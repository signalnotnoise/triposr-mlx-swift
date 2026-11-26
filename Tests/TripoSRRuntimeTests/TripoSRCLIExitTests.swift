import Darwin
import Foundation
import Testing

/// Locates the built `TripoSRCLI` in the same products directory as this test image.
///
/// `xcodebuild test` hosts the bundle under `xctest`, so `CommandLine.arguments[0]`
/// is not the test binary. The products directory is a few levels above the loaded
/// test image: `Debug/TripoSRRuntimeTests.xctest/Contents/MacOS/<image>`.
func builtTripoSRCLI() -> URL? {
    let fileManager = FileManager.default
    var info = Dl_info()
    guard dladdr(#dsohandle, &info) != 0, let fileName = info.dli_fname else {
        return nil
    }
    var directory = URL(fileURLWithPath: String(cString: fileName))
        .resolvingSymlinksInPath()
        .deletingLastPathComponent()
    for _ in 0..<8 {
        let candidate = directory.appendingPathComponent("TripoSRCLI")
        if fileManager.isExecutableFile(atPath: candidate.path) {
            return candidate
        }
        let parent = directory.deletingLastPathComponent()
        if parent.path == directory.path {
            break
        }
        directory = parent
    }
    return nil
}

func runTripoSRCLI(
    arguments: [String],
    workingDirectory: URL,
    outputDirectory: URL
) throws -> (status: Int32, reason: Process.TerminationReason, output: String) {
    var imageInfo = Dl_info()
    let imagePath: String
    if dladdr(#dsohandle, &imageInfo) != 0, let fileName = imageInfo.dli_fname {
        imagePath = String(cString: fileName)
    } else {
        imagePath = "unknown"
    }
    let cli = try #require(
        builtTripoSRCLI(),
        "TripoSRCLI was not found above the test image at \(imagePath)"
    )
    let outputURL = outputDirectory.appendingPathComponent("stdout.txt")
    let errorURL = outputDirectory.appendingPathComponent("stderr.txt")
    FileManager.default.createFile(atPath: outputURL.path, contents: nil)
    FileManager.default.createFile(atPath: errorURL.path, contents: nil)
    let outputHandle = try FileHandle(forWritingTo: outputURL)
    let errorHandle = try FileHandle(forWritingTo: errorURL)
    defer {
        try? outputHandle.close()
        try? errorHandle.close()
    }
    var environment = ProcessInfo.processInfo.environment
    environment.removeValue(forKey: "TRIPOSR_WEIGHTS_PATH")
    let process = Process()
    process.executableURL = cli
    process.arguments = arguments
    process.currentDirectoryURL = workingDirectory
    process.environment = environment
    process.standardOutput = outputHandle
    process.standardError = errorHandle
    try process.run()
    process.waitUntilExit()
    let stdout = (try? String(contentsOf: outputURL, encoding: .utf8)) ?? ""
    let stderr = (try? String(contentsOf: errorURL, encoding: .utf8)) ?? ""
    return (process.terminationStatus, process.terminationReason, stdout + stderr)
}

func makeEmptyCLIWorkspace() throws -> (root: URL, workingDirectory: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("triposr-cli-exit-\(UUID().uuidString)", isDirectory: true)
    let workingDirectory = root
        .appendingPathComponent("empty", isDirectory: true)
        .appendingPathComponent("cwd", isDirectory: true)
    try FileManager.default.createDirectory(
        at: workingDirectory, withIntermediateDirectories: true)
    return (root, workingDirectory)
}

struct TripoSRCLIExitTests {
    @Test func smokeExitsNonzeroWhenWeightsAreMissing() throws {
        let workspace = try makeEmptyCLIWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace.root) }
        let result = try runTripoSRCLI(
            arguments: [],
            workingDirectory: workspace.workingDirectory,
            outputDirectory: workspace.root
        )
        #expect(result.reason == .exit)
        #expect(result.status != 0)
        #expect(result.output.contains("Failed:"))
    }

    @Test func parityExitsNonzeroWhenInputsAreMissing() throws {
        let workspace = try makeEmptyCLIWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace.root) }
        let result = try runTripoSRCLI(
            arguments: ["--parity"],
            workingDirectory: workspace.workingDirectory,
            outputDirectory: workspace.root
        )
        #expect(result.reason == .exit)
        #expect(result.status != 0)
        #expect(result.output.contains("Failed:"))
        #expect(result.output.contains("parity inputs must be supplied"))
    }

    @Test func parityExitsNonzeroWhenWeightsAreMissing() throws {
        let workspace = try makeEmptyCLIWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace.root) }
        let parity = workspace.workingDirectory.appendingPathComponent("parity", isDirectory: true)
        try FileManager.default.createDirectory(at: parity, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: parity.appendingPathComponent("image.npy").path, contents: Data())
        FileManager.default.createFile(
            atPath: parity.appendingPathComponent("positions.npy").path, contents: Data())
        let result = try runTripoSRCLI(
            arguments: ["--parity"],
            workingDirectory: workspace.workingDirectory,
            outputDirectory: workspace.root
        )
        #expect(result.reason == .exit)
        #expect(result.status != 0)
        #expect(result.output.contains("Failed:"))
        #expect(result.output.contains("weights not found"))
    }
}
