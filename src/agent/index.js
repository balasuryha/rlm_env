import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import fs from "fs/promises";
import path from "path";
import pdf from "pdf-parse/lib/pdf-parse.js";
import mammoth from "mammoth";

// Update this to your data folder
const ALLOWED_DIR = "/Users/balasuryhalavakumar/Documents/AI Projects/mcp/data";

const server = new Server(
  { name: "pro-filesystem-mcp", version: "1.2.0" },
  { capabilities: { tools: {} } }
);

/**
 * Validates that the path is inside the allowed directory.
 * Prevents AI from accessing your system files.
 */
async function validatePath(requestedPath) {
  const absolutePath = path.resolve(requestedPath);
  const rootPath = path.resolve(ALLOWED_DIR);
  if (!absolutePath.startsWith(rootPath)) {
    throw new Error(`Access denied: ${requestedPath} is outside the allowed sandbox.`);
  }
  return absolutePath;
}

/**
 * Smart Reader: Checks if path is a directory and handles PDF/Docx conversion.
 */
async function readSmart(filePath) {
  const stats = await fs.stat(filePath);
  
  if (stats.isDirectory()) {
    throw new Error(`Error: "${path.basename(filePath)}" is a directory. Use list_files to see what's inside.`);
  }

  const ext = path.extname(filePath).toLowerCase();
  const dataBuffer = await fs.readFile(filePath);

  if (ext === ".pdf") {
    const data = await pdf(dataBuffer);
    return data.text;
  } 
  if (ext === ".docx") {
    const result = await mammoth.extractRawText({ buffer: dataBuffer });
    return result.value;
  }
  
  return dataBuffer.toString("utf-8");
}

// --- Registering Tools ---

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list_files",
        description: "List files and folders with size and modification date.",
        inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }
      },
      {
        name: "read_file",
        description: "Read text from .txt, .xml, .md, .pdf, or .docx.",
        inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }
      },
      {
        name: "write_file",
        description: "Save text content to a file (.txt, .md, .xml, .json).",
        inputSchema: { 
          type: "object", 
          properties: { path: { type: "string" }, content: { type: "string" } }, 
          required: ["path", "content"] 
        }
      },
      {
        name: "search_files",
        description: "Search for specific text inside all files in a folder.",
        inputSchema: { 
          type: "object", 
          properties: { 
            query: { type: "string", description: "Keyword or phrase to find" },
            directory: { type: "string", description: "Folder to search within" }
          }, 
          required: ["query", "directory"] 
        }
      },
      {
        name: "delete_file",
        description: "Permanently delete a file.",
        inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }
      }
    ]
  };
});

// --- Handling Tool Logic ---

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "list_files": {
        const fullPath = await validatePath(args.path);
        const entries = await fs.readdir(fullPath, { withFileTypes: true });
        const list = await Promise.all(entries.map(async (e) => {
          const stats = await fs.stat(path.join(fullPath, e.name));
          const type = e.isDirectory() ? "[FOLDER]" : "[FILE]";
          return `${type} ${e.name} (${stats.size} bytes, modified: ${stats.mtime.toLocaleString()})`;
        }));
        return { content: [{ type: "text", text: list.join("\n") }] };
      }

      case "read_file": {
        const fullPath = await validatePath(args.path);
        const content = await readSmart(fullPath);
        return { content: [{ type: "text", text: content }] };
      }

      case "write_file": {
        const fullPath = await validatePath(args.path);
        await fs.writeFile(fullPath, args.content, "utf-8");
        return { content: [{ type: "text", text: `Successfully saved to ${args.path}` }] };
      }

      case "delete_file": {
        const fullPath = await validatePath(args.path);
        await fs.unlink(fullPath);
        return { content: [{ type: "text", text: `Deleted ${args.path}` }] };
      }

      case "search_files": {
        const dirPath = await validatePath(args.directory);
        const files = await fs.readdir(dirPath);
        let matches = [];

        for (const file of files) {
          try {
            const filePath = path.join(dirPath, file);
            const content = await readSmart(filePath); // Uses the smart reader to include PDFs
            if (content.toLowerCase().includes(args.query.toLowerCase())) {
              matches.push(file);
            }
          } catch (e) {
            // Skip folders or unreadable files during search
            continue;
          }
        }
        return { 
          content: [{ 
            type: "text", 
            text: matches.length > 0 ? `Found "${args.query}" in: ${matches.join(", ")}` : `No matches found for "${args.query}".` 
          }] 
        };
      }

      default:
        throw new Error("Tool not found");
    }
  } catch (err) {
    return { isError: true, content: [{ type: "text", text: err.message }] };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Pro MCP Server is running...");