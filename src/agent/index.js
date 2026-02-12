import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import fs from "fs/promises";
import path from "path";
import pdf from "pdf-parse/lib/pdf-parse.js";
import mammoth from "mammoth";

const ALLOWED_DIR = "/Users/balasuryhalavakumar/Documents/AI Projects/mcp/data";

const server = new Server(
  { name: "pro-filesystem", version: "1.1.0" },
  { capabilities: { tools: {} } }
);

// --- Advanced Helper Functions ---

async function validatePath(requestedPath) {
  const absolutePath = path.resolve(requestedPath);
  if (!absolutePath.startsWith(path.resolve(ALLOWED_DIR))) {
    throw new Error(`Access denied: ${requestedPath} is outside allowed directory`);
  }
  return absolutePath;
}

// --- Tool Definitions ---

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list_files",
        description: "List files with sizes and modification dates",
        inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }
      },
      {
        name: "read_file",
        description: "Smart read for txt, xml, pdf, docx",
        inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }
      },
      {
        name: "search_files",
        description: "Search for a keyword inside all files in a directory",
        inputSchema: { 
          type: "object", 
          properties: { 
            query: { type: "string", description: "Keyword to find" },
            directory: { type: "string", description: "Where to search" }
          }, 
          required: ["query", "directory"] 
        }
      },
      {
        name: "write_file",
        description: "Write text-based content to a file",
        inputSchema: { 
          type: "object", 
          properties: { path: { type: "string" }, content: { type: "string" } }, 
          required: ["path", "content"] 
        }
      }
    ]
  };
});

// --- Execution Handler ---

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "list_files": {
        const fullPath = await validatePath(args.path);
        const entries = await fs.readdir(fullPath, { withFileTypes: true });
        const details = await Promise.all(entries.map(async (e) => {
            const stats = await fs.stat(path.join(fullPath, e.name));
            return `${e.name} (${stats.size} bytes, mod: ${stats.mtime.toISOString()})`;
        }));
        return { content: [{ type: "text", text: details.join("\n") }] };
      }

      case "read_file": {
        const fullPath = await validatePath(args.path);
        const ext = path.extname(fullPath).toLowerCase();
        const dataBuffer = await fs.readFile(fullPath);
        
        let text;
        if (ext === ".pdf") text = (await pdf(dataBuffer)).text;
        else if (ext === ".docx") text = (await mammoth.extractRawText({ buffer: dataBuffer })).value;
        else text = dataBuffer.toString("utf-8");

        return { content: [{ type: "text", text }] };
      }

      case "search_files": {
        const fullPath = await validatePath(args.directory);
        const files = await fs.readdir(fullPath);
        let results = [];
        
        for (const file of files) {
          const content = await fs.readFile(path.join(fullPath, file), 'utf-8').catch(() => "");
          if (content.includes(args.query)) results.push(file);
        }
        return { content: [{ type: "text", text: `Found "${args.query}" in: ${results.join(", ") || "None"}` }] };
      }

      case "write_file": {
        const fullPath = await validatePath(args.path);
        await fs.writeFile(fullPath, args.content, "utf-8");
        return { content: [{ type: "text", text: `Saved to ${args.path}` }] };
      }

      default:
        throw new Error("Tool not found");
    }
  } catch (error) {
    return { isError: true, content: [{ type: "text", text: error.message }] };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Pro MCP Server active.");