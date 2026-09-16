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

function renderRun(result, scroll = true) {
  byId("result-panel").hidden = false;
  const answerStates = {
    answered: "Đã trả lời",
    partial: "Trả lời một phần",
    insufficient_evidence: "Thiếu bằng chứng",
    model_abstained: "Model chưa trả lời",
  };
  const answerBadge = result.answer
    ? badge(answerStates[result.answer.status] || result.answer.status)
    : badge("Chưa có đáp án");
  answerBadge.classList.add(result.answer?.status || "no_evidence");
  if (result.answer?.status === "model_abstained") {
    answerBadge.style.backgroundColor = "#fde7e2";
    answerBadge.style.color = "#a13b2e";
  }
  const contextBadge = text("span", `Context: ${result.status}`, `badge ${result.status}`);
  byId("result-badges").replaceChildren(badge(result.branch), contextBadge, answerBadge);
  renderAnswer(result.answer);
  byId("run-id").textContent = `Run ID: ${result.run_id}`;
  const warningBox = byId("warnings");
  const warnings = [...result.warnings];
  if (result.answer?.status === "model_abstained") {
    warnings.push(`Gemini chưa tạo được claim có trích dẫn sau ${result.answer.attempts || 1} lần thử; context đã có bằng chứng được chọn.`);
  }
  warningBox.hidden = !warnings.length;
  warningBox.textContent = warnings.join(" · ");
  byId("citations-title").textContent = result.answer?.status === "model_abstained"
    ? "Bằng chứng đã chọn (chưa được dùng trong đáp án)" : "Nguồn trích dẫn";
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
  if (scroll) byId("result-panel").scrollIntoView({behavior: "smooth", block: "start"});
}

let activeChatId = null;
let chatBusy = false;

function setChatBusy(value) {
  chatBusy = value;
  byId("send-chat").disabled = value;
  byId("new-chat").disabled = value;
  byId("rename-chat").disabled = value;
  byId("delete-chat").disabled = value;
}

function renderSessionList(sessions) {
  const list = byId("chat-sessions");
  list.replaceChildren();
  if (!sessions.length) list.append(text("p", "Chưa có phiên nào.", "muted"));
  for (const session of sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chat-session${session.session_id === activeChatId ? " active" : ""}`;
    button.append(text("span", session.title, "chat-session-title"));
    button.append(text("span", `${session.turn_count} lượt hỏi`, "chat-session-meta"));
    button.addEventListener("click", () => {
      if (!chatBusy) openChat(session.session_id).catch((error) => setStatus("ask-status", error.message, true));
    });
    list.append(button);
  }
}

async function refreshSessionList() {
  const data = await request("/api/chats");
  renderSessionList(data.sessions);
  return data.sessions;
}

function showNewChat() {
  activeChatId = null;
  byId("chat-title").textContent = "Cuộc trò chuyện mới";
  byId("rename-chat").hidden = true;
  byId("delete-chat").hidden = true;
  byId("chat-messages").replaceChildren(text("p", "Bắt đầu bằng một câu hỏi về tài liệu. Các lượt trao đổi sẽ được lưu trong phiên này.", "chat-empty"));
  byId("result-panel").hidden = true;
  byId("question").value = "";
  setStatus("ask-status", "");
}

function addAnswerText(container, value, result) {
  for (const part of value.split(/(\[\d+\])/g)) {
    if (/^\[\d+\]$/.test(part) && result) {
      const button = text("button", part, "cite-button");
      button.type = "button";
      button.title = "Xem nguồn trích dẫn";
      button.addEventListener("click", () => {
        renderRun(result);
        const target = byId(`citation-${part.slice(1, -1)}`);
        if (target) target.scrollIntoView({behavior: "smooth", block: "center"});
      });
      container.append(button);
    } else {
      container.append(document.createTextNode(part));
    }
  }
}

function renderConversation(session) {
  activeChatId = session.session_id;
  byId("chat-title").textContent = session.title;
  byId("rename-chat").hidden = false;
  byId("delete-chat").hidden = false;
  const list = byId("chat-messages");
  list.replaceChildren();
  if (!session.messages.length) {
    list.append(text("p", "Bắt đầu bằng một câu hỏi về tài liệu. Các lượt trao đổi sẽ được lưu trong phiên này.", "chat-empty"));
  }
  for (const message of session.messages) {
    const card = document.createElement("article");
    card.className = `chat-message ${message.role}`;
    card.append(text("div", message.role === "user" ? "Bạn" : "CRAG", "chat-message-head"));
    const body = document.createElement("div");
    body.className = "chat-message-body";
    if (message.role === "assistant") addAnswerText(body, message.text, message.result);
    else body.textContent = message.text;
    card.append(body);
    if (message.role === "assistant" && message.result) {
      const actions = document.createElement("div");
      actions.className = "chat-message-actions";
      const source = text("button", `Xem nguồn · ${message.run_id?.slice(0, 8) || "run"}`);
      source.type = "button";
      source.addEventListener("click", () => renderRun(message.result));
      actions.append(source);
      card.append(actions);
    }
    list.append(card);
  }
  list.scrollTop = list.scrollHeight;
}

async function openChat(sessionId) {
  const session = await request(`/api/chats/${sessionId}`);
  renderConversation(session);
  byId("result-panel").hidden = true;
  byId("question").value = "";
  await refreshSessionList();
}

async function initializeChats() {
  const sessions = await refreshSessionList();
  if (sessions.length) await openChat(sessions[0].session_id);
  else showNewChat();
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
  if (chatBusy) return;
  const question = byId("question").value.trim();
  if (!question) return;
  setChatBusy(true);
  setStatus("ask-status", "Đang retrieval, kiểm tra bằng chứng và tạo câu trả lời…");
  try {
    if (!activeChatId) {
      const created = await request("/api/chats", {
        method: "POST", headers: localHeaders, body: "{}",
      });
      activeChatId = created.session_id;
    }
    const session = await request(`/api/chats/${activeChatId}/messages`, {
      method: "POST", headers: localHeaders,
      body: JSON.stringify({
        question,
        candidate_limit: Number(byId("candidate-limit").value),
        limit: Number(byId("limit").value),
        document_id: byId("scope").value || null,
      }),
    });
    renderConversation(session);
    byId("question").value = "";
    const last = session.messages.at(-1);
    if (last?.result) renderRun(last.result, false);
    await refreshSessionList();
    setStatus("ask-status", `Đã lưu vào phiên · ${last?.result?.branch || "CRAG"}`);
  } catch (error) {
    setStatus("ask-status", error.message, true);
    await refreshSessionList().catch(() => {});
  } finally {
    setChatBusy(false);
  }
});

byId("new-chat").addEventListener("click", async () => {
  if (chatBusy) return;
  showNewChat();
  await refreshSessionList().catch((error) => setStatus("ask-status", error.message, true));
  byId("question").focus();
});

byId("rename-chat").addEventListener("click", async () => {
  if (chatBusy || !activeChatId) return;
  const proposed = window.prompt("Tên phiên trò chuyện (tối đa 100 ký tự):", byId("chat-title").textContent);
  if (proposed === null) return;
  setChatBusy(true);
  try {
    const session = await request(`/api/chats/${activeChatId}`, {
      method: "PUT", headers: localHeaders, body: JSON.stringify({title: proposed}),
    });
    renderConversation(session);
    await refreshSessionList();
    setStatus("ask-status", "Đã đổi tên phiên trò chuyện.");
  } catch (error) {
    setStatus("ask-status", error.message, true);
  } finally {
    setChatBusy(false);
  }
});

byId("delete-chat").addEventListener("click", async () => {
  if (chatBusy || !activeChatId || !window.confirm("Xóa phiên trò chuyện và các tin nhắn trong phiên? Các run CRAG đã lưu vẫn được giữ lại.")) return;
  setChatBusy(true);
  try {
    await request(`/api/chats/${activeChatId}`, {method: "DELETE", headers: localHeaders});
    showNewChat();
    const sessions = await refreshSessionList();
    if (sessions.length) await openChat(sessions[0].session_id);
  } catch (error) {
    setStatus("ask-status", error.message, true);
  } finally {
    setChatBusy(false);
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
initializeChats().catch((error) => setStatus("ask-status", error.message, true));
