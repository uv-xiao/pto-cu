export function text(value) {
  return document.createTextNode(String(value));
}

export function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const span = document.createElement("span");
  span.append(text(label));
  const strong = document.createElement("strong");
  strong.append(text(value));
  item.append(span, strong);
  return item;
}

export function paragraph(label, value) {
  const item = document.createElement("p");
  const strong = document.createElement("strong");
  strong.append(text(`${label}: `));
  item.append(strong, text(value));
  return item;
}

export function fieldList(fields) {
  const list = document.createElement("dl");
  list.className = "meta-list";
  fields.forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.append(text(label));
    const detail = document.createElement("dd");
    detail.append(text(value));
    list.append(term, detail);
  });
  return list;
}

export function evidenceList(refs) {
  const list = document.createElement("ul");
  refs.forEach((ref) => {
    const item = document.createElement("li");
    item.append(text(`${ref.path}: ${ref.symbols.join(", ")}`));
    list.append(item);
  });
  return list;
}

export function textList(items) {
  const list = document.createElement("ul");
  items.forEach((value) => {
    const item = document.createElement("li");
    item.append(text(value));
    list.append(item);
  });
  return list;
}

export function namedList(title, items) {
  const heading = document.createElement("h3");
  heading.append(text(title));
  return [heading, textList(items)];
}

export function commandBlock(command) {
  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.append(text(command));
  pre.append(code);
  return pre;
}

export function table(headers, rows) {
  const tableEl = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.append(text(header));
    headerRow.append(th);
  });
  thead.append(headerRow);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.append(text(cell));
      tr.append(td);
    });
    tbody.append(tr);
  });
  tableEl.append(thead, tbody);
  return tableEl;
}
