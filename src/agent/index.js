import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
// 1. Import everything as 'mcpFs'
import * as mcpFs from "@modelcontextprotocol/server-filesystem/dist/index.js";

const allowedDirectories = [
  "/Users/balasuryhalavakumar/Documents/AI_Projects/mcp/data"
];

// 2. Access the class from the namespace
// In most versions, it's mcpFs.FileSystemServer
const fsServer = new mcpFs.default.FileSystemServer(allowedDirectories);

const server = new Server(
  { name: "my-fs-server", version: "1.0.0" },
  { capabilities: fsServer.capabilities }
);

const transport = new StdioServerTransport();
await server.connect(transport);

console.error("MCP Filesystem Server running...");