"use strict";

const byId = (id) => document.getElementById(id);
const localHeaders = {"Content-Type": "application/json", "X-CRAG-Local": "1"};

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function setStatus(id, message, error = false) {
  const element = byId(id);
  element.textContent = message;
  element.classList.toggle("error", error);
}

function text(tag, value, className = "") {
  const element = document.createElement(tag);
  element.textContent = String(value);
  if (className) element.className = className;
  return element;
}

function badge(value) {
  return text("span", value, `badge ${value}`);
}

function renderAnswer(answer) {
  const box = byId("answer-text");
  box.replaceChildren();
  if (!answer) {
    box.textContent = "Lượt chạy cũ chưa có câu trả lời; chỉ có context và citation.";
    return;
  }
  const parts = answer.text.split(/(\[\d+\])/g);
  for (const part of parts) {
    if (/^\[\d+\]$/.test(part)) {
      const button = text("button", part, "cite-button");
      button.type = "button";
      button.addEventListener("click", () => {
        const target = document.getElementById(`citation-${part.slice(1, -1)}`);
        if (target) {
          target.scrollIntoView({behavior: "smooth", block: "center"});
          target.focus({preventScroll: true});
        }
      });
      box.append(button);
    } else {
      box.append(document.createTextNode(part));
    }
  }
}

function renderRun(result) {
  byId("result-panel").hidden = false;
  byId("result-badges").replaceChildren(badge(result.branch), badge(result.status));
  renderAnswer(result.answer);
  byId("run-id").textContent = `Run ID: ${result.run_id}`;
  const warningBox = byId("warnings");
  warningBox.hidden = !result.warnings.length;
  warningBox.textContent = result.warnings.join(" · ");
  const list = byId("citations");
  list.replaceChildren();
  if (!result.citations.length) list.append(text("p", "Không có nguồn trích dẫn.", "muted"));
  for (const citation of result.citations) {
    const card = document.createElement("article");
    card.className = "citation";
    card.id = `citation-${citation.marker.slice(1, -1)}`;
    card.tabIndex = -1;
    const head = document.createElement("div");
    head.className = "citation-head";
    head.append(text("span", `${citation.marker} ${citation.source_type === "internal" ? "Tài liệu" : "Web"}`, "citation-marker"));
    if (citation.viewer_url) {
      const link = text("a", "Mở nguồn ↗");
      link.href = citation.viewer_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      head.append(link);
    }
    card.append(head);
    const pages = citation.metadata.pages;
    const location = Array.isArray(pages) && pages.length ? ` · Trang ${pages.join(", ")}` : "";
    card.append(text("div", `${citation.source_ref}${location}`, "citation-meta"));
    card.append(text("blockquote", citation.text || "Đoạn nguồn không còn trong context."));
    list.append(card);
  }
  byId("result-panel").scrollIntoView({behavior: "smooth", block: "start"});
}

async function loadDocuments() {
  const data = await request("/api/documents");
  const select = byId("scope");
  const selected = select.value;
  select.replaceChildren(new Option("Toàn bộ tài liệu", ""));
  const list = byId("document-list");
  list.replaceChildren();
  if (!data.documents.length) list.append(text("p", "Chưa có tài liệu được index.", "muted"));
  for (const doc of data.documents) {
    const name = doc.metadata?.original_filename || doc.source_path.split(/[\\/]/).pop();
    select.add(new Option(name, doc.document_id));
    const row = document.createElement("article");
    row.className = "document";
    const description = document.createElement("div");
    description.append(text("div", name, "document-title"));
    description.append(text("div", `${doc.chunk_count} chunks · ${doc.document_id} · ${doc.source_path}`, "document-meta"));
    if (doc.index_status && doc.index_status !== "current") {
      const reasons = {
        outdated_pipeline: "bộ xử lý đã thay đổi",
        source_changed: "file gốc đã thay đổi",
        source_missing: "không tìm thấy file gốc",
      };
      description.append(text("div", `Cần làm mới: ${reasons[doc.index_status] || doc.index_status}. Không thể truy vấn tài liệu này cho đến khi làm mới.`, "error"));
    }
    const actions = document.createElement("div");
    actions.className = "document-actions";
    const source = text("a", "Mở nguồn", "secondary");
    source.href = `/api/documents/${doc.document_id}/source`;
    source.target = "_blank";
    source.rel = "noopener noreferrer";
    actions.append(source);
    const refresh = text("button", "Làm mới", "secondary");
    refresh.type = "button";
    refresh.addEventListener("click", () => changeDocument(doc.document_id, "refresh"));
    actions.append(refresh);
    if (doc.managed_upload) {
      const replace = text("button", "Thay file", "secondary");
      replace.type = "button";
      replace.addEventListener("click", () => chooseReplacement(doc.document_id));
      actions.append(replace);
    }
    const remove = text("button", "Xóa", "danger");
    remove.type = "button";
    remove.addEventListener("click", () => changeDocument(doc.document_id, "delete"));
    actions.append(remove);
    row.append(description, actions);
    list.append(row);
  }
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
}

function fileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = () => reject(new Error("Không thể đọc file."));
    reader.readAsDataURL(file);
  });
}

async function upload(file, documentId = null) {
  setStatus("document-status", `Đang xử lý ${file.name}…`);
  try {
    const content_base64 = await fileAsBase64(file);
    const path = documentId ? `/api/documents/${documentId}` : "/api/documents";
    const result = await request(path, {
      method: documentId ? "PUT" : "POST", headers: localHeaders,
      body: JSON.stringify({filename: file.name, content_base64}),
    });
    setStatus("document-status", `${result.status}: ${file.name} · ${result.chunk_count} chunks`);
    await loadDocuments();
  } catch (error) {
    setStatus("document-status", error.message, true);
  }
}

function chooseReplacement(documentId) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".pdf,.docx,.md,.markdown,.txt";
  input.addEventListener("change", () => {
    if (input.files?.[0]) upload(input.files[0], documentId);
  });
  input.click();
}

async function changeDocument(documentId, action) {
  if (action === "delete" && !window.confirm("Xóa tài liệu khỏi Qdrant và xóa bản lưu do CRAG quản lý? File gốc bên ngoài không bị xóa.")) return;
  setStatus("document-status", action === "delete" ? "Đang xóa tài liệu…" : "Đang làm mới tài liệu…");
  try {
    const result = await request(
      action === "delete" ? `/api/documents/${documentId}` : `/api/documents/${documentId}/refresh`,
      {method: action === "delete" ? "DELETE" : "POST", headers: localHeaders},
    );
    setStatus("document-status", action === "delete" ? `Đã xóa ${result.removed_chunks} chunks.` : `Đã làm mới ${result.chunk_count} chunks.`);
    await loadDocuments();
  } catch (error) {
    setStatus("document-status", error.message, true);
  }
}

byId("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  setStatus("ask-status", "Đang retrieval, kiểm tra bằng chứng và tạo câu trả lời…");
  try {
    const data = await request("/api/ask", {
      method: "POST", headers: localHeaders,
      body: JSON.stringify({
        question: byId("question").value,
        candidate_limit: Number(byId("candidate-limit").value),
        limit: Number(byId("limit").value),
        document_id: byId("scope").value || null,
      }),
    });
    renderRun(data);
    setStatus("ask-status", `Hoàn tất · ${data.branch} · ${data.run_id}`);
  } catch (error) {
    setStatus("ask-status", error.message, true);
  } finally {
    button.disabled = false;
  }
});

byId("upload-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const file = byId("upload-file").files?.[0];
  if (file) upload(file);
});
byId("refresh-list").addEventListener("click", () => loadDocuments().catch((error) => setStatus("document-status", error.message, true)));
byId("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    renderRun(await request(`/api/runs/${byId("run-input").value.trim()}`));
  } catch (error) {
    setStatus("ask-status", error.message, true);
  }
});
loadDocuments().catch((error) => setStatus("document-status", error.message, true));
