import Foundation
import MLX
import MLXNN

public class TripoSRRenderer: Module {
    public let radius: Float

    public init(radius: Float = 0.87) {
        self.radius = radius
    }

    /// Bilinear grid sample matching PyTorch `F.grid_sample(..., mode="bilinear", align_corners=False, padding_mode="zeros")`.
    /// `image` shape: [B, H, W, C] (NHWC)
    /// `grid` shape: [B, N, 2] with (x, y) in [-1, 1]
    /// Returns: [B, N, C]
    private func gridSampleBilinear(image: MLXArray, grid: MLXArray) -> MLXArray {
        let B = image.shape[0]
        let H = image.shape[1]
        let W = image.shape[2]
        let C = image.shape[3]
        let N = grid.shape[1]

        // align_corners=False: pixel centers at ((x+1)*W*0.5 - 0.5, ...)
        let gx = (grid[0..., 0..., 0] + 1.0) * (Float(W) * 0.5) - 0.5
        let gy = (grid[0..., 0..., 1] + 1.0) * (Float(H) * 0.5) - 0.5

        let x0 = floor(gx)
        let x1 = x0 + 1
        let y0 = floor(gy)
        let y1 = y0 + 1

        // Unclamped fractional weights (PyTorch zeros-pad uses these with OOB → 0)
        let wa = (x1 - gx) * (y1 - gy)  // (x0,y0)
        let wb = (x1 - gx) * (gy - y0)  // (x0,y1)
        let wc = (gx - x0) * (y1 - gy)  // (x1,y0)
        let wd = (gx - x0) * (gy - y0)  // (x1,y1)

        // Valid masks for zeros padding (use logicalAnd — `&&` is unavailable on MLXArray)
        let in00 = logicalAnd(
            logicalAnd(x0 .>= 0, x0 .< Float(W)),
            logicalAnd(y0 .>= 0, y0 .< Float(H))
        )
        let in01 = logicalAnd(
            logicalAnd(x1 .>= 0, x1 .< Float(W)),
            logicalAnd(y0 .>= 0, y0 .< Float(H))
        )
        let in10 = logicalAnd(
            logicalAnd(x0 .>= 0, x0 .< Float(W)),
            logicalAnd(y1 .>= 0, y1 .< Float(H))
        )
        let in11 = logicalAnd(
            logicalAnd(x1 .>= 0, x1 .< Float(W)),
            logicalAnd(y1 .>= 0, y1 .< Float(H))
        )

        let x0c = maximum(minimum(x0, Float(W - 1)), 0.0)
        let x1c = maximum(minimum(x1, Float(W - 1)), 0.0)
        let y0c = maximum(minimum(y0, Float(H - 1)), 0.0)
        let y1c = maximum(minimum(y1, Float(H - 1)), 0.0)

        // Batched gather: flatten to [B*H*W, C], index = b*H*W + y*W + x
        let flat = image.reshaped([B * H * W, C])
        let batchBase = MLXArray(0..<B).asType(Float.self).reshaped([B, 1]) * Float(H * W)
        let base = MLX.broadcast(batchBase, to: [B, N])

        func gather(_ xs: MLXArray, _ ys: MLXArray, _ valid: MLXArray) -> MLXArray {
            let idx = (base + ys * Float(W) + xs).asType(Int32.self).reshaped([-1])
            let vals = flat.take(idx, axis: 0).reshaped([B, N, C])
            let mask = valid.asType(Float.self).reshaped([B, N, 1])
            return vals * mask
        }

        let v00 = gather(x0c, y0c, in00)
        let v01 = gather(x1c, y0c, in01)
        let v10 = gather(x0c, y1c, in10)
        let v11 = gather(x1c, y1c, in11)

        return v00 * wa.reshaped([B, N, 1])
            + v01 * wc.reshaped([B, N, 1])
            + v10 * wb.reshaped([B, N, 1])
            + v11 * wd.reshaped([B, N, 1])
    }

    public func queryTriplane(
        decoder: NeRFMLP,
        positions: MLXArray, // [N, 3] or [B, N, 3]
        triplanes: MLXArray  // [B, 3, H, W, C]
    ) -> (density: MLXArray, color: MLXArray) {

        let B = triplanes.shape[0]
        let N = positions.shape.reversed()[1]

        // Normalize positions from (-radius, radius) to (-1, 1). Do not clamp:
        // PyTorch grid_sample uses zero padding for coordinates outside that range.
        let posNorm = positions / radius

        // Create 2D projections for the 3 planes: XY, XZ, YZ
        let xy = posNorm.take(MLXArray([0, 1]), axis: -1)
        let xz = posNorm.take(MLXArray([0, 2]), axis: -1)
        let yz = posNorm.take(MLXArray([1, 2]), axis: -1)

        // Expand batch if positions are just [N, 3]
        let gridXY = xy.shape.count == 2 ? MLX.broadcast(xy, to: [B, N, 2]) : xy
        let gridXZ = xz.shape.count == 2 ? MLX.broadcast(xz, to: [B, N, 2]) : xz
        let gridYZ = yz.shape.count == 2 ? MLX.broadcast(yz, to: [B, N, 2]) : yz

        // Sample each plane using custom Bilinear Sampler
        let f0 = gridSampleBilinear(image: triplanes[0..., 0], grid: gridXY)
        let f1 = gridSampleBilinear(image: triplanes[0..., 1], grid: gridXZ)
        let f2 = gridSampleBilinear(image: triplanes[0..., 2], grid: gridYZ)

        // TripoSR uses feature_reduction: "concat"
        let concatFeatures = concatenated([f0, f1, f2], axis: -1) // [B, N, 120]

        // Pass to decoder
        let mlpOut = decoder(concatFeatures)

        // Split output (TripoSR uses trunc_exp for density and sigmoid for color)
        let densityRaw = mlpOut[0..., 0..., 0..<1]
        let colorRaw = mlpOut[0..., 0..., 1...]

        let densityAct = exp(densityRaw - 1.0) // trunc_exp with density_bias = -1.0
        let colorAct = sigmoid(colorRaw)

        return (density: densityAct, color: colorAct)
    }
}
