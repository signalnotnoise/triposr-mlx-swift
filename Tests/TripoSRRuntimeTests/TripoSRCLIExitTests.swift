import Foundation
import Testing

/// Locates the built `TripoSRCLI` next to the test bundle.
///
/// `xcodebuild test` places the executable in the same products directory as
/// the `.xctest` bundle. Walking up from the test process covers that layout
/// and a `.build/` SwiftPM layout.
func builtTripoSRCLI() -> URL? {
    let fileManager = FileManager.default
    var directory = URL(fileURLWithPath: CommandLine.arguments[0])
        .standardizedFileURL
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

struct TripoSRCLIExitTests {
    @Test func smokeExitsNonzeroWhenWeightsAreMissing() throws {
        let cli = try #require(
            builtTripoSRCLI(),
            "TripoSRCLI was not next to the test bundle. Build the TripoSRCLI scheme before testing."
        )

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("triposr-cli-exit-\(UUID().uuidString)", isDirectory: true)
        let workingDirectory = root
            .appendingPathComponent("empty", isDirectory: true)
            .appendingPathComponent("cwd", isDirectory: true)
        try FileManager.default.createDirectory(
            at: workingDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let outputURL = root.appendingPathComponent("stdout.txt")
        let errorURL = root.appendingPathComponent("stderr.txt")
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
        process.currentDirectoryURL = workingDirectory
        process.environment = environment
        process.standardOutput = outputHandle
        process.standardError = errorHandle
        try process.run()
        process.waitUntilExit()

        let output = (try? String(contentsOf: outputURL, encoding: .utf8)) ?? ""
        let errorOutput = (try? String(contentsOf: errorURL, encoding: .utf8)) ?? ""
        let combined = output + errorOutput

        #expect(process.terminationReason == .exit)
        #expect(process.terminationStatus != 0)
        #expect(combined.contains("Failed:"))
    }
}
