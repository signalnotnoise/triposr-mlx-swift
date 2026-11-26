import Foundation
import MLX
import MLXNN

public class DINOv2Model: Module {
    public let patchEmbed: DINOv2PatchEmbed
    public let clsToken: MLXArray
    public let posEmbed: MLXArray

    public let blocks: [DINOv2Layer]
    public let norm: LayerNorm

    public init(
        imgSize: Int = 518,
        patchSize: Int = 14,
        inChannels: Int = 3,
        embedDim: Int = 768,
        depth: Int = 12,
        numHeads: Int = 12,
        mlpRatio: Float = 4.0
    ) {
        self.patchEmbed = DINOv2PatchEmbed(patchSize: patchSize, inChannels: inChannels, embedDim: embedDim)

        // Num patches = (imgSize / patchSize) ^ 2
        let numPatches = (imgSize / patchSize) * (imgSize / patchSize)

        self.clsToken = MLXRandom.normal([1, 1, embedDim]) * 0.02
        self.posEmbed = MLXRandom.normal([1, numPatches + 1, embedDim]) * 0.02

        self.blocks = (0..<depth).map { _ in
            DINOv2Layer(dim: embedDim, numHeads: numHeads, mlpRatio: mlpRatio)
        }
        self.norm = LayerNorm(dimensions: embedDim, eps: 1e-6)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let embedDim = clsToken.shape[2]

        // 1. Patch Embed
        var xOut = patchEmbed(x)
        let numPatches = xOut.shape[1]
        let side = Int(Double(numPatches).squareRoot().rounded())

        let clsTokens = MLX.broadcast(clsToken, to: [B, 1, embedDim])
        xOut = concatenated([clsTokens, xOut], axis: 1)

        // 3. Add positional embeddings (interpolate 14×14→runtime grid when needed)
        let pe = interpolatePosEmbed(numPatches: numPatches, side: side)
        xOut = xOut + pe

        // 4. Transformer Blocks — eval periodically to bound peak Metal memory
        for (i, block) in blocks.enumerated() {
            xOut = block(xOut)
            if i % 3 == 2 || i == blocks.count - 1 {
                eval(xOut)
            }
        }

        // 5. Final Norm
        xOut = norm(xOut)

        return xOut
    }

    /// HF/DINO-style 2D bicubic-ish bilinear resize of patch pos embeds (CLS unchanged).
    private func interpolatePosEmbed(numPatches: Int, side: Int) -> MLXArray {
        let seq = posEmbed.shape[1]
        // Already matches (e.g. 224 → 197 tokens)
        if seq == numPatches + 1 {
            return posEmbed
        }
        let oldPatches = seq - 1
        let oldSide = Int(Double(oldPatches).squareRoot().rounded())
        precondition(oldSide * oldSide == oldPatches, "posEmbed patch count is not square")

        let cls = posEmbed[0..., 0..<1, 0...]  // [1, 1, D]
        var patch = posEmbed[0..., 1..., 0...]  // [1, oldPatches, D]
        let D = patch.shape[2]
        // [1, oldSide, oldSide, D] → NHWC for bilinear via nearest-neighbor upsample grid
        patch = patch.reshaped([1, oldSide, oldSide, D])

        // Build sampling grid for newSide×newSide in [-1,1] (align_corners=False style)
        let newSide = side
        var ys = [Float]()
        var xs = [Float]()
        ys.reserveCapacity(newSide * newSide)
        xs.reserveCapacity(newSide * newSide)
        for iy in 0..<newSide {
            for ix in 0..<newSide {
                // Map pixel centers: (i + 0.5) / new * old - 0.5  → then to [-1,1] for grid_sample
                let fy = (Float(iy) + 0.5) * Float(oldSide) / Float(newSide) - 0.5
                let fx = (Float(ix) + 0.5) * Float(oldSide) / Float(newSide) - 0.5
                // bilinear weights on integer grid
                ys.append(fy)
                xs.append(fx)
            }
        }
        // Manual bilinear over the 2D pos-embed grid
        var out = [Float](repeating: 0, count: newSide * newSide * D)
        let src = patch.asArray(Float.self) // [1*oldSide*oldSide*D] row-major NHWC
        for i in 0..<(newSide * newSide) {
            let fy = ys[i]
            let fx = xs[i]
            let y0 = Int(floor(fy))
            let x0 = Int(floor(fx))
            let y1 = y0 + 1
            let x1 = x0 + 1
            let wy = fy - Float(y0)
            let wx = fx - Float(x0)
            let y0c = min(max(y0, 0), oldSide - 1)
            let y1c = min(max(y1, 0), oldSide - 1)
            let x0c = min(max(x0, 0), oldSide - 1)
            let x1c = min(max(x1, 0), oldSide - 1)
            let w00 = (1 - wy) * (1 - wx)
            let w01 = (1 - wy) * wx
            let w10 = wy * (1 - wx)
            let w11 = wy * wx
            for d in 0..<D {
                let v00 = src[(y0c * oldSide + x0c) * D + d]
                let v01 = src[(y0c * oldSide + x1c) * D + d]
                let v10 = src[(y1c * oldSide + x0c) * D + d]
                let v11 = src[(y1c * oldSide + x1c) * D + d]
                out[i * D + d] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
            }
        }
        let patchNew = MLXArray(out).reshaped([1, numPatches, D])
        return concatenated([cls, patchNew], axis: 1)
    }
}
