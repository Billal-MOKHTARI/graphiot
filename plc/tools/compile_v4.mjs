// Headless STruC++ compile for an OpenPLC Runtime v4 bundle.
//
// Mirrors what the OpenPLC Editor v4 does before uploading (compile() with the
// debug table, generated.hpp header name, project MD5), so the runtime's
// Makefile.strucpp and runtime_v4_entry.cpp shim build it exactly as an
// Editor upload. The Python side (graphiot.plc.bundle) adds defines.h, the
// plugin conf/*.json and zips the directory.
//
// usage: node compile_v4.mjs --st <program.st> --out <dir> --md5 <hex>
// prints a JSON summary on stdout; exits 1 with JSON errors on failure.
import fs from "node:fs";
import path from "node:path";
import {
  compile,
  discoverStlibs,
  findBundledLibsDir,
  findRuntimeIncludeDir,
  getVersion,
} from "strucpp/node";

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0 || i + 1 >= process.argv.length) throw new Error(`missing --${name}`);
  return process.argv[i + 1];
}

const stPath = arg("st");
const outDir = arg("out");
const md5 = arg("md5");
const source = fs.readFileSync(stPath, "utf-8");

const libsDir = findBundledLibsDir();
const libraries = libsDir ? discoverStlibs(libsDir) : [];

const result = compile(source, {
  debug: true,
  lineMapping: true,
  optimizationLevel: 0,
  headerFileName: "generated.hpp",
  fileName: path.basename(stPath),
  md5,
  libraries,
});

if (!result.success) {
  process.stdout.write(JSON.stringify({ ok: false, errors: result.errors }, null, 1));
  process.exit(1);
}

fs.mkdirSync(outDir, { recursive: true });
const written = [];
const put = (name, content) => {
  fs.writeFileSync(path.join(outDir, name), content);
  written.push(name);
};

put("generated.hpp", result.headerCode);
for (const f of result.cppFiles) put(f.name, f.content);
if (result.debugTableCpp) put("generated_debug.cpp", result.debugTableCpp);
if (result.debugMap) put("debug-map.json", JSON.stringify(result.debugMap));

// Ship the header-only runtime the way the Editor does: core/generated/strucpp_runtime/include.
const incSrc = findRuntimeIncludeDir();
const incDst = path.join(outDir, "strucpp_runtime", "include");
fs.mkdirSync(incDst, { recursive: true });
let headers = 0;
for (const f of fs.readdirSync(incSrc)) {
  const p = path.join(incSrc, f);
  if (fs.statSync(p).isFile()) {
    fs.copyFileSync(p, path.join(incDst, f));
    headers += 1;
  }
}

process.stdout.write(JSON.stringify({
  ok: true,
  strucpp: getVersion(),
  files: written,
  runtime_headers: headers,
  warnings: result.warnings.length,
}));
