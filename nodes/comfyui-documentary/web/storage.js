// "Documentary → Stored files": lists what's in ComfyUI's output, input
// (uploaded photos) and temp folders, and deletes the files you pick, one at a
// time or ticked together. Server side: /documentary/storage in __init__.py.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SECTIONS = [
  ["output", "Generated (videos, audio, images)"],
  ["input", "Uploaded photos and files"],
  ["temp", "Temporary files (previews)"],
];

const CSS = `
.doc-store-back{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;justify-content:center}
.doc-store{background:var(--comfy-menu-bg,#202020);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);
  border-radius:10px;width:min(860px,94vw);max-height:88vh;display:flex;flex-direction:column;font:13px system-ui,sans-serif}
.doc-store header,.doc-store footer{padding:12px 16px;display:flex;gap:12px;align-items:center;justify-content:space-between}
.doc-store header{border-bottom:1px solid var(--border-color,#444)}
.doc-store footer{border-top:1px solid var(--border-color,#444)}
.doc-store h2{margin:0;font-size:16px}
.doc-store .body{overflow:auto;padding:4px 16px 12px}
.doc-store h3{margin:14px 0 6px;font-size:13px;display:flex;gap:8px;align-items:center}
.doc-store h3 .sum{opacity:.7;font-weight:normal}
.doc-store table{width:100%;border-collapse:collapse}
.doc-store td{padding:4px 6px;border-bottom:1px solid var(--border-color,#333);vertical-align:middle}
.doc-store td.name{word-break:break-all}
.doc-store td.num{text-align:right;white-space:nowrap;opacity:.8}
.doc-store a{color:inherit}
.doc-store button{background:var(--comfy-input-bg,#333);color:inherit;border:1px solid var(--border-color,#555);
  border-radius:6px;padding:5px 10px;cursor:pointer}
.doc-store button.danger{background:#a33;border-color:#c55;color:#fff}
.doc-store button:disabled{opacity:.4;cursor:default}
.doc-store .empty{opacity:.6;padding:4px 6px}
`;

function fmtSize(n) {
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

function fmtDate(t) {
  return new Date(t * 1000).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" });
}

function viewUrl(f) {
  const i = f.path.lastIndexOf("/");
  const q = new URLSearchParams({
    filename: f.path.slice(i + 1), subfolder: i < 0 ? "" : f.path.slice(0, i), type: f.folder,
  });
  return api.apiURL(`/view?${q}`);
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  e.append(...children);
  return e;
}

async function remove(files) {
  const r = await api.fetchApi("/documentary/storage/delete", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ files }),
  });
  const res = await r.json();
  if (res.errors?.length) alert(`Not deleted:\n${res.errors.join("\n")}`);
  return res;
}

async function openStorage() {
  if (!document.getElementById("doc-store-css")) {
    document.head.append(el("style", { id: "doc-store-css" }, CSS));
  }
  const selected = new Set();
  const key = (f) => `${f.folder}/${f.path}`;
  const back = el("div", { class: "doc-store-back", onclick: (e) => e.target === back && back.remove() });
  const box = el("div", { class: "doc-store" });
  back.append(box);
  document.body.append(back);

  async function render() {
    box.replaceChildren(el("div", { class: "body" }, "Loading…"));
    let data;
    try {
      data = await (await api.fetchApi("/documentary/storage")).json();
    } catch (e) {
      box.replaceChildren(el("div", { class: "body" }, `Could not list files: ${e}`));
      return;
    }
    const byKey = new Map(data.files.map((f) => [key(f), f]));
    for (const k of [...selected]) if (!byKey.has(k)) selected.delete(k);
    const used = data.files.reduce((a, f) => a + f.size, 0);

    const delSelected = el("button", { class: "danger" });
    const updateFooter = () => {
      const size = [...selected].reduce((a, k) => a + byKey.get(k).size, 0);
      delSelected.textContent = `Delete selected (${selected.size}, ${fmtSize(size)})`;
      delSelected.disabled = !selected.size;
    };
    delSelected.addEventListener("click", async () => {
      const files = [...selected].map((k) => byKey.get(k));
      if (!confirm(`Delete ${files.length} file(s), ${fmtSize(files.reduce((a, f) => a + f.size, 0))}? This can't be undone.`)) return;
      const res = await remove(files.map(({ folder, path }) => ({ folder, path })));
      selected.clear();
      app.extensionManager?.toast?.add({ severity: "success", summary: `Deleted ${res.deleted} file(s), freed ${fmtSize(res.freed)}`, life: 4000 });
      render();
    });

    const body = el("div", { class: "body" });
    for (const [folder, title] of SECTIONS) {
      const files = data.files.filter((f) => f.folder === folder);
      const sum = files.reduce((a, f) => a + f.size, 0);
      const all = el("input", { type: "checkbox", title: "Select all in this section", disabled: !files.length });
      all.checked = files.length > 0 && files.every((f) => selected.has(key(f)));
      const boxes = [];
      all.addEventListener("change", () => {
        files.forEach((f) => (all.checked ? selected.add(key(f)) : selected.delete(key(f))));
        boxes.forEach((b) => (b.checked = all.checked));
        updateFooter();
      });
      body.append(el("h3", {}, all, title, el("span", { class: "sum" }, `${files.length} files, ${fmtSize(sum)}`)));
      if (!files.length) {
        body.append(el("div", { class: "empty" }, "Nothing here."));
        continue;
      }
      const table = el("table");
      for (const f of files) {
        const cb = el("input", { type: "checkbox" });
        cb.checked = selected.has(key(f));
        cb.addEventListener("change", () => {
          cb.checked ? selected.add(key(f)) : selected.delete(key(f));
          all.checked = files.every((x) => selected.has(key(x)));
          updateFooter();
        });
        boxes.push(cb);
        const del = el("button", {
          title: "Delete this file",
          onclick: async () => {
            if (!confirm(`Delete ${f.folder}/${f.path} (${fmtSize(f.size)})? This can't be undone.`)) return;
            await remove([{ folder: f.folder, path: f.path }]);
            render();
          },
        }, "Delete");
        table.append(el("tr", {},
          el("td", {}, cb),
          el("td", { class: "name" }, el("a", { href: viewUrl(f), target: "_blank", rel: "noopener" }, f.path)),
          el("td", { class: "num" }, fmtSize(f.size)),
          el("td", { class: "num" }, fmtDate(f.mtime)),
          el("td", {}, del)));
      }
      body.append(table);
    }

    box.replaceChildren(
      el("header", {},
        el("h2", {}, "Stored files"),
        el("span", {}, `In these folders: ${fmtSize(used)} · free on the volume: ${fmtSize(data.disk.free)} of ${fmtSize(data.disk.total)}`),
        el("button", { onclick: () => back.remove() }, "Close")),
      body,
      el("footer", {},
        el("button", { onclick: render }, "Refresh"),
        delSelected));
    updateFooter();
  }
  render();
}

app.registerExtension({
  name: "documentary.storage",
  commands: [{ id: "documentary.storage", label: "Stored files", icon: "pi pi-trash", function: openStorage }],
  menuCommands: [{ path: ["Documentary"], commands: ["documentary.storage"] }],
});
