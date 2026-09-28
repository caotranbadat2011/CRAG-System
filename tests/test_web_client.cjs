const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

// Exercise the actual browser request helper without starting the UI or an API.
const app = fs.readFileSync(path.join(__dirname, "../src/crag_ingestion/static/app.js"), "utf8");
const helper = app.slice(0, app.indexOf("\nfunction setStatus"));
function request(response) {
  const context = vm.createContext({fetch: async () => response});
  vm.runInContext(helper, context);
  return vm.runInContext('request("/api/chats")', context);
}

test("successful JSON remains usable", async () => {
  const data = await request(new Response('{"sessions":[]}', {headers: {"Content-Type":"application/json; charset=utf-8"}}));
  assert.equal(data.sessions.length, 0);
});
test("HTML errors expose HTTP status instead of a JSON syntax error", async () => {
  await assert.rejects(request(new Response("<!DOCTYPE HTML><html></html>", {
    status: 501, headers: {"Content-Type":"text/html"},
  })), /HTTP 501.*không phải JSON/);
});
test("malformed JSON has an actionable message", async () => {
  await assert.rejects(request(new Response('{"unfinished":', {
    headers: {"Content-Type":"application/json"},
  })), /JSON bị lỗi hoặc chưa đầy đủ/);
});
test("JSON API errors retain the server explanation", async () => {
  await assert.rejects(request(new Response('{"error":"quota exhausted"}', {
    status: 503, headers: {"Content-Type":"application/json"},
  })), /quota exhausted/);
});
