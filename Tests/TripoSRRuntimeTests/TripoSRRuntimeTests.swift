import Foundation
import MLX
import Testing

@testable import TripoSRRuntime

/// Published weights are not part of the package. CI and a fresh checkout
/// leave this nil, and the weight-loading test is skipped.
func tripoSRPublishedWeightsURL() -> URL? {
    if let path = ProcessInfo.processInfo.environment["TRIPOSR_WEIGHTS_PATH"], !path.isEmpty,
        FileManager.default.fileExists(atPath: path)
    {
        return URL(fileURLWithPath: path)
    }
    let relative = [
        "triposr_swift_ready.safetensors",
        "triposr_weights/triposr_swift_ready.safetensors",
    ]
    for name in relative {
        let url = URL(fileURLWithPath: name).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
    }
    return nil
}

/// Deterministic unit tests for the TripoSR Swift runtime's public contracts.
///
/// These tests use small synthetic tensors and random module parameters.
/// They check shapes, value ranges, and failure behavior. They do not
/// compare against PyTorch, and they do not extract a mesh. The published
/// weights file is optional: that test is skipped when the file is absent.
struct TripoSRRuntimeTests {

    init() {
        // Plain `swift test` has no MLX metallib; force CPU like the parity CLI.
        Device.setDefault(device: Device(.cpu))
    }

    @Test func preprocessImageAppliesImageNetMeanAndStd() {
        let zeros = MLXArray.zeros([1, 1, 1, 3])
        let normalized = TripoSRModel.preprocessImage(zeros)
        let values = normalized.asArray(Float.self)
        #expect(values.count == 3)
        for channel in 0..<3 {
            let expected = -TripoSRModel.imageNetMean[channel] / TripoSRModel.imageNetStd[channel]
            #expect(abs(values[channel] - expected) < 1e-6)
        }

        let ones = MLXArray.ones([1, 2, 2, 3])
        let normalizedOnes = TripoSRModel.preprocessImage(ones)
        #expect(normalizedOnes.shape == [1, 2, 2, 3])
        let firstPixel = Array(normalizedOnes.asArray(Float.self).prefix(3))
        for channel in 0..<3 {
            let expected =
                (1 - TripoSRModel.imageNetMean[channel]) / TripoSRModel.imageNetStd[channel]
            #expect(abs(firstPixel[channel] - expected) < 1e-6)
        }
    }

    @Test func tokenizerTokenizeDetokenizeRoundTrip() {
        let tokenizer = Triplane1DTokenizer(planeSize: 4, numChannels: 8)
        let tokens = tokenizer(batchSize: 2)
        #expect(tokens.shape == [2, 3 * 4 * 4, 8])

        let planes = tokenizer.detokenize(tokens)
        #expect(planes.shape == [2, 3, 4, 4, 8])

        // Detokenize is a pure reshape: batch element 0 must reproduce the
        // learnable embeddings exactly.
        let roundTripped = planes[0].asArray(Float.self)
        let embeddings = tokenizer.embeddings.asArray(Float.self)
        #expect(roundTripped == embeddings)
    }

    @Test func rendererQueryShapesAndActivationRanges() {
        let renderer = TripoSRRenderer(radius: 0.87)
        let decoder = NeRFMLP(inChannels: 120, nNeurons: 8, nHiddenLayers: 2)
        let triplanes = MLXArray.zeros([1, 3, 4, 4, 40])
        // Includes an out-of-radius point: zero padding must keep the query
        // finite instead of clamping or trapping.
        let positions = MLXArray(
            [Float(0), 0, 0, 0.5, -0.5, 0.25, 2.0, 2.0, 2.0], [3, 3])

        let (density, color) = renderer.queryTriplane(
            decoder: decoder, positions: positions, triplanes: triplanes)

        #expect(density.shape == [1, 3, 1])
        #expect(color.shape == [1, 3, 3])
        // Density uses trunc_exp (strictly positive); color uses sigmoid.
        #expect(density.asArray(Float.self).allSatisfy { $0 > 0 && $0.isFinite })
        #expect(color.asArray(Float.self).allSatisfy { $0 > 0 && $0 < 1 })
    }

    @Test func loadWeightsRejectsMissingFile() {
        let model = TripoSRModel()
        let missing = URL(fileURLWithPath: "/nonexistent/triposr_swift_ready.safetensors")
        #expect(throws: (any Error).self) {
            try model.loadWeights(from: missing)
        }
    }

    /// Loads https://huggingface.co/signalNotNoise/triposr-mlx
    /// `triposr_swift_ready.safetensors` when a local copy is present.
    /// Absent file: skipped, not failed.
    @Test(
        .enabled(
            if: tripoSRPublishedWeightsURL() != nil,
            "triposr_swift_ready.safetensors is absent"
        )
    )
    func loadWeightsFromPublishedFileWhenPresent() throws {
        let url = try #require(tripoSRPublishedWeightsURL())
        let model = TripoSRModel()
        try model.loadWeights(from: url)
    }
}
