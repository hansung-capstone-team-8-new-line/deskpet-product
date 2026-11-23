import Foundation

struct ModelConfig: Codable {
    var backend: String
    var role: String
    var device: String
    var paths: [String: String]
    var library: String?
    var contextTokens: Int?
    var maxTokens: Int?
    var threads: Int?
    var streaming: Bool?
}

struct JetsonConfig: Codable {
    var library: String
    var harnessEnvFile: String?
    var hardwareSocket: String?
    var routerTimeout: Double?
    var outputTimeout: Double?
    var sttTimeout: Double?
    var ttsTimeout: Double?
    var routeConfidenceThreshold: Double?
    var sttConfidenceThreshold: Double?
    var routerContextEnabled: Bool?
    var directLLM: Bool?
    var shortOutput: Bool?
    var performanceDirectory: String?
    var startupReminder: Bool?
    var startupReminderText: String?
    var models: [String: ModelConfig]
}

// MARK: - Minimal ONNX Protobuf INT4 / FP16 Weight Converter (zero Python dependencies)

private enum OnnxQuantizer {
    static let minElemsInt4 = 4096
    static let minElemsFp16 = 1024
    static let blockSize = 32

    static let tensorProtoFloat: UInt64 = 1
    static let tensorProtoFloat16: UInt64 = 10
    static let tensorProtoInt4: UInt64 = 22

    @inline(__always)
    static func float16BitsToFloat(_ bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exp = UInt32((bits >> 10) & 0x1F)
        let mant = UInt32(bits & 0x03FF)
        if exp == 0 {
            if mant == 0 { return Float(bitPattern: sign) }
            var e = Int32(127 - 14)
            var m = mant << 13
            while (m & 0x0080_0000) == 0 {
                m <<= 1
                e -= 1
            }
            m &= 0x007F_FFFF
            return Float(bitPattern: sign | (UInt32(e) << 23) | m)
        } else if exp == 31 {
            return Float(bitPattern: sign | 0x7F80_0000 | (mant != 0 ? 0x0040_0000 : 0))
        }
        let fExp = (exp + 127 - 15) << 23
        return Float(bitPattern: sign | fExp | (mant << 13))
    }

    @inline(__always)
    static func floatToFloat16Bits(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exp = Int((bits >> 23) & 0xFF)
        let mant = bits & 0x007F_FFFF

        if exp == 0xFF {
            return mant != 0 ? (sign | 0x7E00) : (sign | 0x7C00)
        }
        let unbiasedExp = exp - 127 + 15
        if unbiasedExp >= 31 {
            return sign | 0x7BFF
        }
        if unbiasedExp <= 0 {
            if unbiasedExp < -10 { return sign }
            let m = mant | 0x0080_0000
            let shift = 14 - unbiasedExp
            let half = m >> shift
            let rem = m & ((1 << shift) - 1)
            let mid = UInt32(1) << (shift - 1)
            let roundUp = (rem > mid) || (rem == mid && (half & 1) == 1)
            return sign | UInt16(half + (roundUp ? 1 : 0))
        }
        var halfExp = UInt16(unbiasedExp)
        var halfMant = UInt16(mant >> 13)
        let rem = mant & 0x1FFF
        if rem > 0x1000 || (rem == 0x1000 && (halfMant & 1) == 1) {
            halfMant += 1
            if halfMant == 0x0400 {
                halfMant = 0
                halfExp += 1
                if halfExp >= 31 {
                    return sign | 0x7BFF
                }
            }
        }
        return sign | (halfExp << 10) | halfMant
    }

    static func readVarint(_ data: Data, _ pos: inout Int) throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while pos < data.endIndex {
            let b = UInt64(data[pos])
            pos += 1
            result |= (b & 0x7F) << shift
            if (b & 0x80) == 0 { return result }
            shift += 7
            if shift >= 64 {
                throw NSError(domain: "OnnxQuantizer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Varint overflow"])
            }
        }
        throw NSError(domain: "OnnxQuantizer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Truncated varint"])
    }

    static func appendVarint(_ out: inout Data, _ v: UInt64) {
        var val = v
        while val >= 0x80 {
            out.append(UInt8((val & 0x7F) | 0x80))
            val >>= 7
        }
        out.append(UInt8(val))
    }

    static func appendSignedVarint(_ out: inout Data, _ v: Int64) {
        appendVarint(&out, UInt64(bitPattern: v))
    }

    static func appendTag(_ out: inout Data, field: UInt64, wire: UInt64) {
        appendVarint(&out, (field << 3) | wire)
    }

    static func appendBytesField(_ out: inout Data, field: UInt64, payload: Data) {
        appendTag(&out, field: field, wire: 2)
        appendVarint(&out, UInt64(payload.count))
        out.append(payload)
    }

    static func appendStringField(_ out: inout Data, field: UInt64, text: String) {
        appendBytesField(&out, field: field, payload: Data(text.utf8))
    }

    static func skipField(_ data: Data, _ pos: inout Int, wire: UInt64) throws -> Range<Int> {
        let start = pos
        switch wire {
        case 0:
            _ = try readVarint(data, &pos)
        case 1:
            guard pos + 8 <= data.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 3) }
            pos += 8
        case 2:
            let len = Int(try readVarint(data, &pos))
            guard len >= 0, pos + len <= data.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 4) }
            pos += len
        case 5:
            guard pos + 4 <= data.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 5) }
            pos += 4
        default:
            throw NSError(domain: "OnnxQuantizer", code: 6, userInfo: [NSLocalizedDescriptionKey: "Unsupported wire type \(wire)"])
        }
        return start..<pos
    }

    static func makeIntAttr(name: String, value: Int64) -> Data {
        var attr = Data()
        appendStringField(&attr, field: 1, text: name)
        appendTag(&attr, field: 3, wire: 0)
        appendSignedVarint(&attr, value)
        appendTag(&attr, field: 20, wire: 0)
        appendVarint(&attr, 2) // AttributeProto.INT = 2
        return attr
    }

    static func makeCastNode(input: String, output: String, nodeName: String) -> Data {
        var node = Data()
        appendStringField(&node, field: 1, text: input)
        appendStringField(&node, field: 2, text: output)
        appendStringField(&node, field: 3, text: nodeName)
        appendStringField(&node, field: 4, text: "Cast")
        appendBytesField(&node, field: 5, payload: makeIntAttr(name: "to", value: Int64(tensorProtoFloat)))
        return node
    }

    static func makeDequantizeInt4Node(input: String, scale: String, output: String,
                                       nodeName: String, axis: Int64, blockSize: Int64?) -> Data {
        var node = Data()
        appendStringField(&node, field: 1, text: input)
        appendStringField(&node, field: 1, text: scale)
        appendStringField(&node, field: 2, text: output)
        appendStringField(&node, field: 3, text: nodeName)
        appendStringField(&node, field: 4, text: "DequantizeLinear")
        appendBytesField(&node, field: 5, payload: makeIntAttr(name: "axis", value: axis))
        if let blockSize {
            appendBytesField(&node, field: 5, payload: makeIntAttr(name: "block_size", value: blockSize))
        }
        return node
    }

    static func buildTensorProto(dims: [Int64], dataType: UInt64, name: String, rawData: Data) -> Data {
        var t = Data()
        for d in dims {
            appendTag(&t, field: 1, wire: 0)
            appendSignedVarint(&t, d)
        }
        appendTag(&t, field: 2, wire: 0)
        appendVarint(&t, dataType)
        if !name.isEmpty {
            appendStringField(&t, field: 8, text: name)
        }
        appendBytesField(&t, field: 9, payload: rawData)
        return t
    }

    enum ConvertedTensor {
        case fp16(tensor: Data, origName: String, rawName: String)
        case int4(tensor: Data, scaleTensor: Data, origName: String, rawName: String,
                  scaleName: String, axis: Int64, blockSize: Int64?)
    }

    static func tryConvertTensorProto(_ tensorData: Data, allowInt4: Bool) throws -> ConvertedTensor? {
        var pos = tensorData.startIndex
        var dims: [Int64] = []
        var dataType: UInt64 = 0
        var dataLocation: UInt64 = 0
        var name: String?
        var rawRange: Range<Int>?
        var floatRanges: [(Range<Int>, Bool)] = []
        var otherFields: [(UInt64, UInt64, Range<Int>)] = []

        while pos < tensorData.endIndex {
            let tag = try readVarint(tensorData, &pos)
            let field = tag >> 3
            let wire = tag & 0x7
            if field == 1 && wire == 0 {
                dims.append(Int64(bitPattern: try readVarint(tensorData, &pos)))
            } else if field == 1 && wire == 2 {
                let len = Int(try readVarint(tensorData, &pos))
                let end = pos + len
                guard end <= tensorData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 7) }
                while pos < end {
                    dims.append(Int64(bitPattern: try readVarint(tensorData, &pos)))
                }
            } else if field == 2 && wire == 0 {
                dataType = try readVarint(tensorData, &pos)
            } else if field == 14 && wire == 0 {
                dataLocation = try readVarint(tensorData, &pos)
            } else if field == 8 && wire == 2 {
                let len = Int(try readVarint(tensorData, &pos))
                guard pos + len <= tensorData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 8) }
                name = String(decoding: tensorData[pos..<(pos + len)], as: UTF8.self)
                pos += len
            } else if field == 9 && wire == 2 {
                let len = Int(try readVarint(tensorData, &pos))
                guard pos + len <= tensorData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 9) }
                rawRange = pos..<(pos + len)
                pos += len
            } else if field == 4 && (wire == 2 || wire == 5) {
                if wire == 2 {
                    let len = Int(try readVarint(tensorData, &pos))
                    guard pos + len <= tensorData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 10) }
                    floatRanges.append((pos..<(pos + len), true))
                    pos += len
                } else {
                    guard pos + 4 <= tensorData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 11) }
                    floatRanges.append((pos..<(pos + 4), false))
                    pos += 4
                }
            } else {
                let r = try skipField(tensorData, &pos, wire: wire)
                otherFields.append((field, wire, r))
            }
        }

        guard dataLocation == 0,
              (dataType == tensorProtoFloat || (allowInt4 && dataType == tensorProtoFloat16)) else { return nil }

        let isInputFp16 = (dataType == tensorProtoFloat16)
        let elemSize = isInputFp16 ? 2 : 4
        let totalRawBytes: Int
        if let rawRange {
            totalRawBytes = rawRange.count
        } else if !isInputFp16 {
            totalRawBytes = floatRanges.reduce(0) { $0 + $1.0.count }
        } else {
            return nil
        }
        let minElems = isInputFp16 ? minElemsInt4 : minElemsFp16
        guard totalRawBytes >= minElems * elemSize, totalRawBytes % elemSize == 0 else { return nil }

        let elemCount = totalRawBytes / elemSize
        var floats = [Float](repeating: 0, count: elemCount)
        var outIndex = 0
        if isInputFp16, let rawRange {
            tensorData[rawRange].withUnsafeBytes { inRaw in
                let inPtr = inRaw.bindMemory(to: UInt16.self)
                for i in 0..<inPtr.count {
                    floats[i] = float16BitsToFloat(UInt16(littleEndian: inPtr[i]))
                }
            }
        } else {
            func copySlice(_ r: Range<Int>) {
                tensorData[r].withUnsafeBytes { inRaw in
                    let inPtr = inRaw.bindMemory(to: UInt32.self)
                    for i in 0..<inPtr.count {
                        floats[outIndex] = Float(bitPattern: UInt32(littleEndian: inPtr[i]))
                        outIndex += 1
                    }
                }
            }
            if let rawRange {
                copySlice(rawRange)
            } else {
                for (r, _) in floatRanges { copySlice(r) }
            }
        }

        let origName = name ?? ""
        let dimProd = dims.reduce(1) { $0 * Int($1) }

        if allowInt4, !origName.isEmpty, dims.count >= 2, dimProd == elemCount, elemCount >= minElemsInt4 {
            let lastAxis = dims.count - 1
            let axis: Int
            let chosenBlock: Int?
            if Int(dims[lastAxis]) % blockSize == 0 {
                axis = lastAxis
                chosenBlock = blockSize
            } else if Int(dims[0]) % blockSize == 0 {
                axis = 0
                chosenBlock = blockSize
            } else {
                axis = 0
                chosenBlock = nil
            }

            let outer = dims[..<axis].reduce(1) { $0 * Int($1) }
            let axisLen = Int(dims[axis])
            let inner = dims[(axis + 1)...].reduce(1) { $0 * Int($1) }

            var scaleDims = dims
            let scalesPerAxis: Int
            let groupLen: Int
            if let chosenBlock {
                scalesPerAxis = axisLen / chosenBlock
                scaleDims[axis] = Int64(scalesPerAxis)
                groupLen = chosenBlock
            } else {
                scalesPerAxis = 1
                scaleDims = [Int64(dims[0])]
                groupLen = axisLen
            }

            let numScales = chosenBlock != nil ? (outer * scalesPerAxis * inner) : Int(dims[0])
            var scales = [Float](repeating: 1.0, count: numScales)

            if let chosenBlock {
                for o in 0..<outer {
                    for sb in 0..<scalesPerAxis {
                        let aStart = sb * chosenBlock
                        for inn in 0..<inner {
                            var maxAbs: Float = 0
                            for k in 0..<chosenBlock {
                                let idx = ((o * axisLen) + (aStart + k)) * inner + inn
                                let a = abs(floats[idx])
                                if a > maxAbs { maxAbs = a }
                            }
                            let s = maxAbs > 0 ? (maxAbs / 7.0) : 1.0
                            scales[(o * scalesPerAxis + sb) * inner + inn] = s
                        }
                    }
                }
            } else {
                let rowStride = elemCount / Int(dims[0])
                for r in 0..<Int(dims[0]) {
                    var maxAbs: Float = 0
                    let base = r * rowStride
                    for k in 0..<rowStride {
                        let a = abs(floats[base + k])
                        if a > maxAbs { maxAbs = a }
                    }
                    scales[r] = maxAbs > 0 ? (maxAbs / 7.0) : 1.0
                }
            }

            let packedCount = (elemCount + 1) / 2
            var packed = Data(count: packedCount)
            packed.withUnsafeMutableBytes { rawOut in
                let outPtr = rawOut.bindMemory(to: UInt8.self)
                for idx in 0..<elemCount {
                    let scaleVal: Float
                    if let chosenBlock {
                        let inn = idx % inner
                        let rem = idx / inner
                        let a = rem % axisLen
                        let o = rem / axisLen
                        let sb = a / chosenBlock
                        scaleVal = scales[(o * scalesPerAxis + sb) * inner + inn]
                    } else {
                        let rowStride = elemCount / Int(dims[0])
                        scaleVal = scales[idx / rowStride]
                    }
                    let qf = (floats[idx] / scaleVal).rounded()
                    let qi = max(-7, min(7, Int(qf)))
                    let nibble = UInt8(bitPattern: Int8(qi)) & 0x0F
                    let byteIdx = idx >> 1
                    if (idx & 1) == 0 {
                        outPtr[byteIdx] = nibble
                    } else {
                        outPtr[byteIdx] |= (nibble << 4)
                    }
                }
            }

            let scaleType: UInt64 = isInputFp16 ? tensorProtoFloat16 : tensorProtoFloat
            var scaleBytes: Data
            if isInputFp16 {
                scaleBytes = Data(count: numScales * 2)
                scaleBytes.withUnsafeMutableBytes { rawScale in
                    let ptr = rawScale.bindMemory(to: UInt16.self)
                    for i in 0..<numScales {
                        ptr[i] = floatToFloat16Bits(scales[i]).littleEndian
                    }
                }
            } else {
                scaleBytes = Data(count: numScales * 4)
                scaleBytes.withUnsafeMutableBytes { rawScale in
                    let ptr = rawScale.bindMemory(to: UInt32.self)
                    for i in 0..<numScales {
                        ptr[i] = scales[i].bitPattern.littleEndian
                    }
                }
            }

            let rawName = "\(origName)__int4_raw"
            let scaleName = "\(origName)__int4_scale"
            let qTensor = buildTensorProto(dims: dims, dataType: tensorProtoInt4, name: rawName, rawData: packed)
            let sTensor = buildTensorProto(dims: scaleDims, dataType: scaleType, name: scaleName, rawData: scaleBytes)
            _ = groupLen
            return .int4(
                tensor: qTensor,
                scaleTensor: sTensor,
                origName: origName,
                rawName: rawName,
                scaleName: scaleName,
                axis: Int64(axis),
                blockSize: chosenBlock.map(Int64.init)
            )
        }

        // Fallback: FP16 storage + Cast(to=FLOAT)
        if isInputFp16 { return nil }
        let rawName = origName.isEmpty ? "" : "\(origName)__fp16_raw"
        var fp16Bytes = Data(count: elemCount * 2)
        fp16Bytes.withUnsafeMutableBytes { outRaw in
            let outPtr = outRaw.bindMemory(to: UInt16.self)
            for i in 0..<elemCount {
                outPtr[i] = floatToFloat16Bits(floats[i]).littleEndian
            }
        }
        let rebuilt = buildTensorProto(dims: dims, dataType: tensorProtoFloat16, name: rawName, rawData: fp16Bytes)
        return .fp16(tensor: rebuilt, origName: origName, rawName: rawName)
    }

    static func rewriteValueInfoIfRenamed(_ viData: Data, renamed: [String: (String, UInt64)]) throws -> Data? {
        var pos = viData.startIndex
        var name: String?
        var typeRange: Range<Int>?
        var others: [(UInt64, UInt64, Range<Int>)] = []
        while pos < viData.endIndex {
            let tag = try readVarint(viData, &pos)
            let field = tag >> 3
            let wire = tag & 0x7
            if field == 1 && wire == 2 {
                let len = Int(try readVarint(viData, &pos))
                guard pos + len <= viData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 12) }
                name = String(decoding: viData[pos..<(pos + len)], as: UTF8.self)
                pos += len
            } else if field == 2 && wire == 2 {
                let len = Int(try readVarint(viData, &pos))
                guard pos + len <= viData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 13) }
                typeRange = pos..<(pos + len)
                pos += len
            } else {
                others.append((field, wire, try skipField(viData, &pos, wire: wire)))
            }
        }
        guard let name, let (newName, newType) = renamed[name] else { return nil }
        var out = Data()
        appendStringField(&out, field: 1, text: newName)
        if let typeRange {
            let updatedType = try rewriteTypeProtoElemType(viData[typeRange], to: newType)
            appendBytesField(&out, field: 2, payload: updatedType)
        }
        for (f, w, r) in others {
            appendTag(&out, field: f, wire: w)
            out.append(viData[r])
        }
        return out
    }

    static func rewriteTypeProtoElemType(_ typeData: Data, to elemType: UInt64) throws -> Data {
        var pos = typeData.startIndex
        var out = Data()
        while pos < typeData.endIndex {
            let tag = try readVarint(typeData, &pos)
            let field = tag >> 3
            let wire = tag & 0x7
            if field == 1 && wire == 2 {
                let len = Int(try readVarint(typeData, &pos))
                guard pos + len <= typeData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 14) }
                let tensorType = try rewriteTensorTypeElemType(typeData[pos..<(pos + len)], to: elemType)
                appendBytesField(&out, field: 1, payload: tensorType)
                pos += len
            } else {
                let r = try skipField(typeData, &pos, wire: wire)
                appendTag(&out, field: field, wire: wire)
                out.append(typeData[r])
            }
        }
        return out
    }

    static func rewriteTensorTypeElemType(_ ttData: Data, to elemType: UInt64) throws -> Data {
        var pos = ttData.startIndex
        var out = Data()
        var wroteElem = false
        while pos < ttData.endIndex {
            let tag = try readVarint(ttData, &pos)
            let field = tag >> 3
            let wire = tag & 0x7
            if field == 1 && wire == 0 {
                _ = try readVarint(ttData, &pos)
                appendTag(&out, field: 1, wire: 0)
                appendVarint(&out, elemType)
                wroteElem = true
            } else {
                let r = try skipField(ttData, &pos, wire: wire)
                appendTag(&out, field: field, wire: wire)
                out.append(ttData[r])
            }
        }
        if !wroteElem {
            appendTag(&out, field: 1, wire: 0)
            appendVarint(&out, elemType)
        }
        return out
    }

    static func rewriteOpsetIdIfDefault(_ opsetData: Data, minVersion: UInt64) throws -> Data {
        var pos = opsetData.startIndex
        var domain = ""
        var version: UInt64 = 0
        var others: [(UInt64, UInt64, Range<Int>)] = []
        while pos < opsetData.endIndex {
            let tag = try readVarint(opsetData, &pos)
            let field = tag >> 3
            let wire = tag & 0x7
            if field == 1 && wire == 2 {
                let len = Int(try readVarint(opsetData, &pos))
                guard pos + len <= opsetData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 15) }
                domain = String(decoding: opsetData[pos..<(pos + len)], as: UTF8.self)
                pos += len
            } else if field == 2 && wire == 0 {
                version = try readVarint(opsetData, &pos)
            } else {
                others.append((field, wire, try skipField(opsetData, &pos, wire: wire)))
            }
        }
        let targetVersion = (domain.isEmpty || domain == "ai.onnx") ? max(version, minVersion) : version
        var out = Data()
        if !domain.isEmpty { appendStringField(&out, field: 1, text: domain) }
        appendTag(&out, field: 2, wire: 0)
        appendVarint(&out, targetVersion)
        for (f, w, r) in others {
            appendTag(&out, field: f, wire: w)
            out.append(opsetData[r])
        }
        return out
    }

    static func convertGraphProto(_ graphData: Data, allowInt4: Bool) throws -> (Data, Int, Int, Bool) {
        var pos = graphData.startIndex
        var nodeRanges: [Range<Int>] = []
        var initRanges: [Range<Int>] = []
        var inputRanges: [Range<Int>] = []
        var otherFields: [(UInt64, UInt64, Range<Int>)] = []

        while pos < graphData.endIndex {
            let tag = try readVarint(graphData, &pos)
            let field = tag >> 3
            let wire = tag & 0x7
            if field == 1 && wire == 2 {
                let len = Int(try readVarint(graphData, &pos))
                guard pos + len <= graphData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 16) }
                nodeRanges.append(pos..<(pos + len))
                pos += len
            } else if field == 5 && wire == 2 {
                let len = Int(try readVarint(graphData, &pos))
                guard pos + len <= graphData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 17) }
                initRanges.append(pos..<(pos + len))
                pos += len
            } else if field == 11 && wire == 2 {
                let len = Int(try readVarint(graphData, &pos))
                guard pos + len <= graphData.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 18) }
                inputRanges.append(pos..<(pos + len))
                pos += len
            } else {
                otherFields.append((field, wire, try skipField(graphData, &pos, wire: wire)))
            }
        }

        var out = Data()
        var renamed: [String: (String, UInt64)] = [:]
        var prependedNodes: [Data] = []
        var int4Count = 0
        var fp16Count = 0

        for (idx, r) in initRanges.enumerated() {
            if let conv = try tryConvertTensorProto(graphData[r], allowInt4: allowInt4) {
                switch conv {
                case let .int4(tensor, scaleTensor, origName, rawName, scaleName, axis, blockSize):
                    renamed[origName] = (rawName, tensorProtoInt4)
                    appendBytesField(&out, field: 5, payload: tensor)
                    appendBytesField(&out, field: 5, payload: scaleTensor)
                    prependedNodes.append(makeDequantizeInt4Node(
                        input: rawName, scale: scaleName, output: origName,
                        nodeName: "deskpet_int4_dq_\(idx)", axis: axis, blockSize: blockSize
                    ))
                    int4Count += 1
                case let .fp16(tensor, origName, rawName) where !origName.isEmpty:
                    renamed[origName] = (rawName, tensorProtoFloat16)
                    appendBytesField(&out, field: 5, payload: tensor)
                    prependedNodes.append(makeCastNode(
                        input: rawName, output: origName, nodeName: "deskpet_fp16_cast_\(idx)"
                    ))
                    fp16Count += 1
                default:
                    appendBytesField(&out, field: 5, payload: graphData[r])
                }
            } else {
                appendBytesField(&out, field: 5, payload: graphData[r])
            }
        }

        for node in prependedNodes {
            appendBytesField(&out, field: 1, payload: node)
        }
        for r in nodeRanges {
            appendBytesField(&out, field: 1, payload: graphData[r])
        }

        for r in inputRanges {
            if !renamed.isEmpty, let rewritten = try rewriteValueInfoIfRenamed(graphData[r], renamed: renamed) {
                appendBytesField(&out, field: 11, payload: rewritten)
            } else {
                appendBytesField(&out, field: 11, payload: graphData[r])
            }
        }

        for (f, w, r) in otherFields {
            appendTag(&out, field: f, wire: w)
            out.append(graphData[r])
        }
        return (out, int4Count, fp16Count, int4Count > 0)
    }

    static func convertModelFile(srcPath: String, dstPath: String, allowInt4: Bool) throws {
        let srcURL = URL(fileURLWithPath: srcPath)
        let dstURL = URL(fileURLWithPath: dstPath)
        let files = FileManager.default
        try files.createDirectory(at: dstURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let data = try Data(contentsOf: srcURL, options: .mappedIfSafe)
        var pos = data.startIndex
        var irVersion: UInt64 = 0
        var opsets: [Data] = []
        var graphData: Data?
        var others: [(UInt64, UInt64, Range<Int>)] = []
        var int4Count = 0
        var fp16Count = 0
        var usedInt4 = false

        while pos < data.endIndex {
            let tag = try readVarint(data, &pos)
            let field = tag >> 3
            let wire = tag & 0x7
            if field == 1 && wire == 0 {
                irVersion = try readVarint(data, &pos)
            } else if field == 8 && wire == 2 {
                let len = Int(try readVarint(data, &pos))
                guard pos + len <= data.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 19) }
                opsets.append(data[pos..<(pos + len)])
                pos += len
            } else if field == 7 && wire == 2 {
                let len = Int(try readVarint(data, &pos))
                guard pos + len <= data.endIndex else { throw NSError(domain: "OnnxQuantizer", code: 20) }
                let (newGraph, c4, c16, u4) = try convertGraphProto(data[pos..<(pos + len)], allowInt4: allowInt4)
                graphData = newGraph
                int4Count += c4
                fp16Count += c16
                usedInt4 = usedInt4 || u4
                pos += len
            } else {
                others.append((field, wire, try skipField(data, &pos, wire: wire)))
            }
        }

        var out = Data()
        let outIr = usedInt4 ? max(irVersion, 10) : irVersion
        if outIr > 0 {
            appendTag(&out, field: 1, wire: 0)
            appendVarint(&out, outIr)
        }
        for op in opsets {
            let updated = usedInt4 ? try rewriteOpsetIdIfDefault(op, minVersion: 21) : op
            appendBytesField(&out, field: 8, payload: updated)
        }
        if let graphData {
            appendBytesField(&out, field: 7, payload: graphData)
        }
        for (f, w, r) in others {
            appendTag(&out, field: f, wire: w)
            out.append(data[r])
        }

        let partURL = URL(fileURLWithPath: dstPath + ".part")
        try out.write(to: partURL)
        if files.fileExists(atPath: dstPath) {
            try files.removeItem(at: dstURL)
        }
        try files.moveItem(at: partURL, to: dstURL)
        print("[configure_jetson_json.swift] Quantized: \(srcPath) -> \(dstPath) (INT4=\(int4Count), FP16=\(fp16Count), \(out.count / (1024 * 1024)) MiB)")
    }
}

// MARK: - Configure Models/native.jetson.local.json

let args = CommandLine.arguments
let useAudioGpu = args.contains("--use-audio-gpu") || ProcessInfo.processInfo.environment["DESKPET_USE_AUDIO_GPU"] == "1"
let lowMemory = args.contains("--low-memory") || ProcessInfo.processInfo.environment["DESKPET_LOW_MEMORY"] == "1"
let configPath = "Models/native.jetson.local.json"
let files = FileManager.default

guard files.fileExists(atPath: configPath),
      let data = files.contents(atPath: configPath) else {
    fputs("[configure_jetson_json.swift] Failed to read \(configPath)\n", stderr)
    exit(1)
}

var cfg = try JSONDecoder().decode(JetsonConfig.self, from: data)
cfg.library = "../.build/native/libdeskpet_inference.so"
cfg.harnessEnvFile = "../../deskpet-harness/.env"

if var router = cfg.models["router"] {
    router.backend = "llama.cpp"
    router.device = "cuda"
    router.library = nil
    router.paths = [
        "model": "router/backbone-merged-q4_0.gguf",
        "heads": "router/heads-f16.gguf",
        "config": "router/router_config.json",
        "chat_template": "router/tokenizer/chat_template.jinja",
    ]
    if router.contextTokens == nil || router.contextTokens == 0 { router.contextTokens = 2560 }
    if router.maxTokens == nil || router.maxTokens == 0 { router.maxTokens = 128 }
    if router.threads == nil || router.threads == 0 { router.threads = 2 }
    cfg.models["router"] = router
}

if var output = cfg.models["output"] {
    output.backend = "llama.cpp"
    output.device = "cuda"
    output.library = nil
    output.paths = [
        "model": "output/qwen-2b-q4_0.gguf",
    ]
    if output.contextTokens == nil || output.contextTokens == 0 { output.contextTokens = 2048 }
    if output.maxTokens == nil || output.maxTokens == 0 { output.maxTokens = 128 }
    if output.threads == nil || output.threads == 0 { output.threads = 2 }
    cfg.models["output"] = output
}

for role in ["vad", "emotion"] {
    if var model = cfg.models[role] {
        model.device = "cuda"
        model.library = nil
        cfg.models[role] = model
    }
}

func firstExistingRelative(in subdirs: [String], candidates: [String]) -> String? {
    for dir in subdirs {
        for name in candidates {
            let rel = dir + "/" + name
            if files.fileExists(atPath: "Models/" + rel) {
                return rel
            }
        }
    }
    return nil
}

func firstMatchingFile(in dir: String, prefix: String, suffixes: [String]) -> String? {
    guard let entries = try? files.contentsOfDirectory(atPath: "Models/" + dir) else { return nil }
    for suffix in suffixes {
        if let hit = entries.sorted().first(where: { $0.hasPrefix(prefix) && $0.hasSuffix(suffix) }) {
            return dir + "/" + hit
        }
    }
    return nil
}

func matchingDirs(under parent: String, prefix: String) -> [String] {
    guard let entries = try? files.contentsOfDirectory(atPath: "Models/" + parent) else { return [] }
    return entries.filter { $0.lowercased().hasPrefix(prefix) }.sorted().map { parent + "/" + $0 }
}

func isUpToDate(_ dstPath: String, relativeTo srcPath: String) -> Bool {
    guard let dstAttrs = try? files.attributesOfItem(atPath: dstPath),
          let dstSize = dstAttrs[.size] as? NSNumber, dstSize.int64Value > 0,
          let dstDate = dstAttrs[.modificationDate] as? Date,
          let srcAttrs = try? files.attributesOfItem(atPath: srcPath),
          let srcDate = srcAttrs[.modificationDate] as? Date else {
        return false
    }
    return dstDate >= srcDate
}

let audioDevice = "cuda"

// MARK: - STT Configuration (Option 1: Zipformer-Korean 72M INT8, Option 2 --low-memory: Zipformer-Korean 174M INT8)
if var stt = cfg.models["stt"] {
    stt.backend = "sherpa-onnx"
    stt.device = audioDevice
    stt.library = nil
    stt.contextTokens = 0
    stt.maxTokens = 0
    if (stt.threads ?? 0) <= 0 { stt.threads = 4 }

    if lowMemory {
        let sttDirs = matchingDirs(under: "stt", prefix: "icefall-asr-ko-streaming-zipformer")
            + matchingDirs(under: "stt", prefix: "sherpa-onnx-streaming-zipformer-korean")
            + ["stt/icefall-asr-ko-streaming-zipformer-174m"]
        let baseDir = sttDirs.first(where: { files.fileExists(atPath: "Models/" + $0 + "/tokens.txt") })
            ?? "stt/icefall-asr-ko-streaming-zipformer-174m"
        let enc = firstMatchingFile(in: baseDir, prefix: "encoder", suffixes: ["chunk-64-left-128.int8.onnx", ".int8.onnx", ".onnx"])
            ?? "\(baseDir)/encoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
        let dec = firstMatchingFile(in: baseDir, prefix: "decoder", suffixes: ["chunk-64-left-128.int8.onnx", ".int8.onnx", ".onnx"])
            ?? "\(baseDir)/decoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
        let joi = firstMatchingFile(in: baseDir, prefix: "joiner", suffixes: ["chunk-64-left-128.int8.onnx", ".int8.onnx", ".onnx"])
            ?? "\(baseDir)/joiner-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
        stt.streaming = true
        stt.paths = [
            "encoder": enc,
            "decoder": dec,
            "joiner": joi,
            "tokens": baseDir + "/tokens.txt",
        ]
    } else {
        let sttDirs = matchingDirs(under: "stt", prefix: "sherpa-onnx-zipformer-korean")
            + ["stt/sherpa-onnx-zipformer-korean-2024-06-24"]
        let baseDir = sttDirs.first(where: { files.fileExists(atPath: "Models/" + $0 + "/tokens.txt") })
            ?? "stt/sherpa-onnx-zipformer-korean-2024-06-24"
        let enc = firstMatchingFile(in: baseDir, prefix: "encoder", suffixes: [".int8.onnx", ".onnx"])
            ?? "\(baseDir)/encoder-epoch-99-avg-1.int8.onnx"
        let dec = firstMatchingFile(in: baseDir, prefix: "decoder", suffixes: [".int8.onnx", ".onnx"])
            ?? "\(baseDir)/decoder-epoch-99-avg-1.onnx"
        let joi = firstMatchingFile(in: baseDir, prefix: "joiner", suffixes: [".int8.onnx", ".onnx"])
            ?? "\(baseDir)/joiner-epoch-99-avg-1.int8.onnx"
        stt.streaming = false
        stt.paths = [
            "encoder": enc,
            "decoder": dec,
            "joiner": joi,
            "tokens": baseDir + "/tokens.txt",
        ]
    }
    cfg.models["stt"] = stt
}

// MARK: - TTS Configuration (Option 1: Qwen3-TTS 0.6B INT4 on CUDA, Option 2 --low-memory: MeloTTS-Korean INT8 + optional Korean BERT INT8 on CUDA)
if var tts = cfg.models["tts"] {
    tts.backend = "onnxruntime"
    tts.device = "cuda"
    tts.library = nil

    if lowMemory {
        let meloDirs = matchingDirs(under: "tts", prefix: "melotts")
            + ["tts/melotts-kr-onnx"]
        let baseDir = meloDirs.first(where: { files.fileExists(atPath: "Models/" + $0 + "/symbols.json") })
            ?? "tts/melotts-kr-onnx"
        let modelRel = firstExistingRelative(in: [baseDir], candidates: ["melotts_kr_fp16.onnx", "melotts_kr_fp32.onnx", "model.onnx", "melotts_kr_int8.onnx", "model.int8.onnx"])
            ?? "\(baseDir)/melotts_kr_fp16.onnx"
        var paths: [String: String] = [
            "model": modelRel,
            "symbols": baseDir + "/symbols.json",
        ]
        if let bertRel = firstExistingRelative(in: [baseDir], candidates: ["bert_kor.int8.onnx", "bert.int8.onnx", "bert_kor.onnx", "bert.onnx"]),
           let bertTokRel = firstExistingRelative(in: [baseDir], candidates: ["bert_tokenizer.json", "tokenizer.json"]) {
            paths["bert"] = bertRel
            paths["bert_tokenizer"] = bertTokRel
        }
        tts.contextTokens = 0
        tts.maxTokens = 0
        if (tts.threads ?? 0) <= 0 { tts.threads = 2 }
        tts.paths = paths
    } else {
        let ttsDirs = matchingDirs(under: "tts", prefix: "qwen3-tts") + ["tts/qwen3-tts-12hz-0.6b-customvoice"]
        let baseDir = ttsDirs.first(where: { files.fileExists(atPath: "Models/" + $0 + "/config.json") })
            ?? "tts/qwen3-tts-12hz-0.6b-customvoice"

        let int4Subdir = "cuda_int4"
        for name in ["text_embed.onnx", "codec_embed.onnx", "residual_embed.onnx", "talker_cache.onnx", "talker.onnx", "code_predictor.onnx"] {
            let srcPath = "Models/\(baseDir)/\(name)"
            let dstPath = "Models/\(baseDir)/\(int4Subdir)/\(name)"
            if files.fileExists(atPath: srcPath), !isUpToDate(dstPath, relativeTo: srcPath) {
                try OnnxQuantizer.convertModelFile(srcPath: srcPath, dstPath: dstPath, allowInt4: true)
            }
        }

        let precisionDirs = [baseDir + "/cuda_int4", baseDir + "/cpu_int4", baseDir + "/cuda_fp16", baseDir + "/cpu_fp16", baseDir]
        let decoderDirs = [baseDir, baseDir + "/cuda_fp32", baseDir + "/cpu_fp32"]
        func ttsArt(_ candidates: [String], in dirs: [String] = precisionDirs, fallback: String) -> String {
            firstExistingRelative(in: dirs, candidates: candidates) ?? (baseDir + "/" + fallback)
        }
        var paths: [String: String] = [
            "text_embed": ttsArt(["text_embed.int4.onnx", "text_embed.onnx"], fallback: "text_embed.onnx"),
            "codec_embed": ttsArt(["codec_embed.int4.onnx", "codec_embed.onnx"], fallback: "codec_embed.onnx"),
            "residual_embed": ttsArt(["residual_embed.int4.onnx", "residual_embed.onnx"], fallback: "residual_embed.onnx"),
            "talker": ttsArt(["talker_cache.int4.onnx", "talker.int4.onnx", "talker_cache.onnx", "talker.onnx"], fallback: "talker_cache.onnx"),
            "code_predictor": ttsArt(["code_predictor.int4.onnx", "code_predictor.onnx"], fallback: "code_predictor.onnx"),
            "tok_decoder": ttsArt(["tok_decoder.onnx"], in: decoderDirs, fallback: "tok_decoder.onnx"),
            "config": baseDir + "/config.json",
        ]
        if let tokJson = firstExistingRelative(in: [baseDir], candidates: ["tokenizer.json"]) {
            paths["tokenizer"] = tokJson
        } else {
            paths["vocab"] = baseDir + "/vocab.json"
            paths["merges"] = baseDir + "/merges.txt"
        }
        tts.maxTokens = (tts.maxTokens == nil || tts.maxTokens == 0) ? 2048 : tts.maxTokens
        if (tts.threads ?? 0) <= 0 { tts.threads = 2 }
        tts.paths = paths
    }
    cfg.models["tts"] = tts
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
var outData = try encoder.encode(cfg)
outData.append(0x0A)
try outData.write(to: URL(fileURLWithPath: configPath))
