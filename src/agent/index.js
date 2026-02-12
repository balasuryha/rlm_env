import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { 
  FileSystemServer, 
  FullTextSearchIndex 
} from "@modelcontextprotocol/server-filesystem";

// 1. Define the directories you want to grant access to
// Use absolute paths for best results
const allowedDirectories = [
  "/Users/yourname/Documents/project-folder",
  "./data" 
];

// 2. Initialize the Filesystem Server
const fsServer = new FileSystemServer(allowedDirectories);

// 3. Connect it to the MCP Transport (usually Stdio)
const server = new Server(
  { name: "my-filesystem-server", version: "1.0.0" },
  { capabilities: fsServer.capabilities }
);

const transport = new StdioServerTransport();
await server.connect(transport);

console.error("MCP Filesystem Server running...");