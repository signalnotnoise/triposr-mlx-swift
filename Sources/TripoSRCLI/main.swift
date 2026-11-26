import Foundation
import MLX
import MLXNN
import TripoSRRuntime

// MARK: - Minimal .npy (float32, C-order) writer for parity dumps

func writeNpy(path: URL, array: MLXArray) throws {
    eval(array)
    let shape = array.shape
    let floats = array.asArray(Float.self)
    var headerBody = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
    if shape.isEmpty {
        headerBody += ")"
    } else if shape.count == 1 {
        headerBody += "\(shape[0]),)"
    } else {
        headerBody += shape.map(String.init).joined(separator: ", ") + ")"
    }
    headerBody += ", }"
    // Pad so (10 + header.count) % 16 == 0
    let prefix = 10
    var header = headerBody
    let pad = (16 - ((prefix + header.utf8.count + 1) % 16)) % 16
    header += String(repeating: " ", count: pad) + "\n"
    let hlen = header.utf8.count

    var raw = Data(count: floats.count * MemoryLayout<Float>.size)
    raw.withUnsafeMutableBytes { dst in
        floats.withUnsafeBufferPointer { src in
            guard let d = dst.baseAddress, let s = src.baseAddress else { return }
            d.copyMemory(from: s, byteCount: floats.count * MemoryLayout<Float>.size)
        }
    }

    var data = Data()
    data.append(0x93)
    data.append(contentsOf: Array("NUMPY".utf8))
    data.append(contentsOf: [0x01, 0x00])
    data.append(UInt8(hlen & 0xFF))
    data.append(UInt8((hlen >> 8) & 0xFF))
    data.append(contentsOf: Array(header.utf8))
    data.append(raw)
    try data.write(to: path)
}

func loadNpyFloat32(path: URL) throws -> (shape: [Int], data: [Float]) {
    let data = try Data(contentsOf: path)
    guard data.count > 10 else { throw NSError(domain: "npy", code: 1) }
    // skip magic/version
    let hlen = Int(data[8]) | (Int(data[9]) << 8)
    let headerStart = 10
    let headerEnd = headerStart + hlen
    let header = String(data: data.subdata(in: headerStart..<headerEnd), encoding: .ascii) ?? ""
    // parse shape from "shape": (a, b, ...)
    guard let shapeRange = header.range(of: "shape': (") ?? header.range(of: "shape\": (") else {
        throw NSError(domain: "npy", code: 2, userInfo: [NSLocalizedDescriptionKey: header])
    }
    let after = header[shapeRange.upperBound...]
    guard let close = after.firstIndex(of: ")") else { throw NSError(domain: "npy", code: 3) }
    let inner = after[..<close]
        .replacingOccurrences(of: " ", with: "")
        .trimmingCharacters(in: CharacterSet(charactersIn: ","))
    let shape: [Int]
    if inner.isEmpty {
        shape = []
    } else {
        shape = inner.split(separator: ",").compactMap { Int($0) }
    }
    let payload = data.subdata(in: headerEnd..<data.count)
    let count = payload.count / 4
    var floats = [Float](repeating: 0, count: count)
    _ = floats.withUnsafeMutableBytes { dst in
        payload.copyBytes(to: dst)
    }
    return (shape, floats)
}

func loadNpzViaPythonFallback() {
    // not used — we use pre-extracted npy from dump script
}

enum CLIError: Error, CustomStringConvertible {
    case weightsNotFound

    var description: String {
        switch self {
        case .weightsNotFound:
            return "weights not found (set TRIPOSR_WEIGHTS_PATH or place triposr_swift_ready.safetensors in the working directory)"
        }
    }
}

func resolveWeights() -> URL? {
    var candidates: [URL] = []
    if let path = ProcessInfo.processInfo.environment["TRIPOSR_WEIGHTS_PATH"], !path.isEmpty {
        candidates.append(URL(fileURLWithPath: path))
    }
    candidates.append(contentsOf: [
        URL(fileURLWithPath: "triposr_swift_ready.safetensors"),
        URL(fileURLWithPath: "triposr_weights/triposr_swift_ready.safetensors"),
        URL(fileURLWithPath: "../triposr_weights/triposr_swift_ready.safetensors"),
        URL(fileURLWithPath: "TripoSR/triposr_weights/triposr_swift_ready.safetensors"),
        URL(fileURLWithPath: "../../TripoSR/triposr_weights/triposr_swift_ready.safetensors"),
    ])
    return candidates.map(\.standardizedFileURL).first {
        FileManager.default.fileExists(atPath: $0.path)
    }
}

func resolveParityDir() -> URL {
    let candidates = [
        URL(fileURLWithPath: "parity", isDirectory: true),
        URL(fileURLWithPath: "../parity", isDirectory: true),
        URL(fileURLWithPath: "TripoSR/parity", isDirectory: true),
        URL(fileURLWithPath: "../../TripoSR/parity", isDirectory: true),
    ]
    for c in candidates {
        let u = c.standardizedFileURL
        if FileManager.default.fileExists(atPath: u.path) { return u }
    }
    let u = URL(fileURLWithPath: "parity", isDirectory: true).standardizedFileURL
    try? FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
    return u
}

func runParityDump() throws {
    // Force global CPU default (TaskLocal alone still left LRM on GPU → timeout).
    Device.setDefault(device: Device(.cpu))
    Memory.cacheLimit = 256 * 1024 * 1024

    let parityDir = resolveParityDir()
    print("Parity dir: \(parityDir.path)")
    print("Device: \(Device.defaultDevice())")

    let imageNpy = parityDir.appendingPathComponent("image.npy")
    let posNpy = parityDir.appendingPathComponent("positions.npy")
    guard FileManager.default.fileExists(atPath: imageNpy.path),
          FileManager.default.fileExists(atPath: posNpy.path) else {
        print("""
        Missing parity inputs. Generate them first:

          source .mlx_venv/bin/activate
          python TripoSR/scripts/dump_python_parity.py
        """)
        return
    }

    let (imgShape, imgData) = try loadNpyFloat32(path: imageNpy)
    let (posShape, posData) = try loadNpyFloat32(path: posNpy)
    print("Loaded image \(imgShape), positions \(posShape)")

    let image = MLXArray(imgData).reshaped(imgShape)
    let positions = MLXArray(posData).reshaped(posShape)

    guard let weightsPath = resolveWeights() else {
        print("Weights not found (triposr_swift_ready.safetensors)")
        return
    }
    print("Weights: \(weightsPath.path)")

    let model = TripoSRModel()
    MLX.eval(model.parameters())
    try model.loadWeights(from: weightsPath)
    print("Weights loaded")

    // Stage 1: DINO
    print("Stage 1/4: DINO image tokens...")
    let imageTokens = model.imageTokenizer(TripoSRModel.preprocessImage(image))
    eval(imageTokens)
    print("  image_tokens \(imageTokens.shape)")
    try writeNpy(path: parityDir.appendingPathComponent("swift_image_tokens.npy"), array: imageTokens)
    print("  wrote swift_image_tokens.npy")

    // Stage 2: LRM backbone (slow on CPU; eval every block already)
    print("Stage 2/4: LRM backbone (CPU; may take several minutes)...")
    let batchSize = image.shape[0]
    var latent = model.tokenizer(batchSize: batchSize)
    eval(latent)
    latent = model.backbone(latent, encoderHiddenStates: imageTokens)
    eval(latent)
    print("  backbone \(latent.shape)")
    try writeNpy(path: parityDir.appendingPathComponent("swift_backbone.npy"), array: latent)
    print("  wrote swift_backbone.npy")

    // Stage 3: triplanes
    print("Stage 3/4: detokenize + upsample...")
    let detok = model.tokenizer.detokenize(latent)
    let triplanes = model.postProcessor(detok)
    eval(triplanes)
    print("  triplanes \(triplanes.shape)")
    try writeNpy(path: parityDir.appendingPathComponent("swift_triplanes.npy"), array: triplanes)
    print("  wrote swift_triplanes.npy")

    // Stage 4: density/color
    print("Stage 4/4: query density/color...")
    let (density, color) = model.evaluate(triplanes: triplanes, positions: positions)
    eval(density, color)
    print("  density \(density.shape) color \(color.shape)")
    try writeNpy(path: parityDir.appendingPathComponent("swift_density.npy"), array: density)
    try writeNpy(path: parityDir.appendingPathComponent("swift_color.npy"), array: color)

    print("Wrote swift_*.npy to \(parityDir.path)")
    print("Next: python TripoSR/scripts/compare_parity.py")
}

func runSmoke() async throws {
    guard let weightsPath = resolveWeights() else {
        throw CLIError.weightsNotFound
    }
    Memory.cacheLimit = 512 * 1024 * 1024
    print("Initializing TripoSR MLX Pipeline (smoke)...")
    let model = TripoSRModel()
    MLX.eval(model.parameters())
    print("Loading weights from: \(weightsPath.path)")
    try model.loadWeights(from: weightsPath)
    print("Weights loaded successfully!")

    let dummyImage = MLXRandom.uniform(low: 0.0, high: 1.0, [1, 512, 512, 3])
    print("Running forward pass (Image -> Triplanes)...")
    let triplanes = model.forwardTriplanes(image: dummyImage)
    MLX.eval(triplanes)
    print("Triplanes: \(triplanes.shape)")

    let dummyPositions = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1000, 3])
    let (density, color) = model.evaluate(triplanes: triplanes, positions: dummyPositions)
    MLX.eval(density, color)
    print("Density bounds: min=\(density.min().item(Float.self)), max=\(density.max().item(Float.self))")
    print("Color bounds: min=\(color.min().item(Float.self)), max=\(color.max().item(Float.self))")
    print("Pipeline executed successfully!")
}

func main() async throws {
    let args = CommandLine.arguments
    if args.contains("--parity") {
        try runParityDump()
    } else {
        try await runSmoke()
    }
}

do {
    try await main()
} catch {
    let message = "Failed: \(error)\n"
    FileHandle.standardError.write(Data(message.utf8))
    exit(1)
}
