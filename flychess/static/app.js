/* flychess UI: board on the left, the fly's brain in the middle, the readout on the right. */
(() => {
  const $ = (s) => document.querySelector(s);
  const game = new Chess();
  let board, thinking = false, history = [];

  // ---------------------------------------------------------------- board
  function humanColor() { return $("#color").value; }
  function onDragStart(_src, piece) {
    if (thinking || game.game_over()) return false;
    if (game.turn() !== humanColor()) return false;
    return piece[0] === humanColor();
  }
  function onDrop(src, dst) {
    const mv = game.move({ from: src, to: dst, promotion: "q" });
    if (!mv) return "snapback";
    afterMove(mv);
    setTimeout(flyMove, 120);
  }
  function onSnapEnd() { board.position(game.fen()); }
  board = Chessboard("board", {
    draggable: true, position: "start", onDragStart, onDrop, onSnapEnd,
    pieceTheme: "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
  });
  window.addEventListener("resize", () => { board.resize(); drawArrows(lastCands); });

  // ---------------------------------------------------------------- state
  let lastCands = [];
  function afterMove(mv) {
    history.push(game.fen());
    board.position(game.fen());
    highlight(mv.from, mv.to);
    renderMoves();
    updateStatus();
  }
  function highlight(from, to) {
    document.querySelectorAll("#board .square-55d63").forEach(e => e.classList.remove("highlight-sq"));
    if (from) $(`#board .square-${from}`)?.classList.add("highlight-sq");
    if (to) $(`#board .square-${to}`)?.classList.add("highlight-sq");
  }
  function renderMoves() {
    const h = game.history(); const ol = $("#moves"); ol.innerHTML = "";
    for (let i = 0; i < h.length; i += 2) {
      const li = document.createElement("li");
      li.innerHTML = `<span>${h[i]}</span> ${h[i + 1] ?? ""}`;
      ol.appendChild(li);
    }
    ol.scrollTop = ol.scrollHeight;
  }
  function updateStatus(extra) {
    const st = $("#status"); st.className = "status";
    if (game.game_over()) {
      st.classList.add("over");
      if (game.in_checkmate()) st.textContent = (game.turn() === humanColor() ? "The fly checkmated you." : "You checkmated the fly!");
      else if (game.in_stalemate()) st.textContent = "Stalemate.";
      else if (game.in_threefold_repetition()) st.textContent = "Draw by repetition.";
      else if (game.insufficient_material()) st.textContent = "Draw: insufficient material.";
      else st.textContent = "Draw.";
      return;
    }
    if (thinking) { st.classList.add("thinking"); st.innerHTML = `<span class="pulse"></span> The fly is looking at the board…`; return; }
    st.textContent = extra || (game.in_check() ? "Your move. You are in check." : "Your move.");
  }

  // ---------------------------------------------------------------- fly
  let scene = null;
  const sceneReady = () => window.flySceneReady?.then(s => { scene = s; return s; }).catch(() => null);
  document.addEventListener("DOMContentLoaded", sceneReady);
  const T_MS = 150, SLOWDOWN = 18;                 // 1 ms of brain time shown over 18 ms

  async function flyMove() {
    if (game.game_over() || game.turn() === humanColor()) { updateStatus(); return; }
    if (!scene) await sceneReady();
    thinking = true; updateStatus(); scene?.reset(); scene?.setThinking(true);
    $("#tl").style.width = "0"; $("#tl-label").textContent = "0 ms";
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/move`);
    let finished = false;
    ws.onopen = () => ws.send(JSON.stringify({ fen: game.fen(), variant: $("#variant").value,
      avoid_repetition: $("#norep").checked, history, slowdown: SLOWDOWN }));
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.error) { finish(); updateStatus("Server error: " + m.error); return; }
      if (m.n) {
        if (m.n.length) scene?.spike(m.n);
        $("#tl").style.width = Math.min(100, 100 * m.t / T_MS) + "%"; $("#tl-label").textContent = m.t.toFixed(0) + " ms";
        return;
      }
      if (m.done) {
        finished = true;
        renderStats(m.stats); renderCands(m.candidates, m.move);
        setTimeout(() => {
          const mv = game.move(m.move, { sloppy: true });
          finish(); afterMove(mv);
          if (!game.game_over()) updateStatus(`The fly played ${m.san}. Your move.`);
        }, 350);
      }
    };
    ws.onerror = () => { if (!finished) { finish(); updateStatus("Connection to the fly failed."); } };
    ws.onclose = () => { if (!finished && thinking) { finish(); updateStatus("Connection closed before the fly moved."); } };
    function finish() { thinking = false; scene?.setThinking(false); }
  }

  function renderStats(s) {
    $("#s-driven").textContent = s.driven; $("#s-neurons").textContent = s.neurons.toLocaleString();
    $("#s-spikes").textContent = s.spikes.toLocaleString(); $("#s-dn").textContent = s.dn_spikes;
    $("#s-time").textContent = s.sim_seconds + " s";
    const cl = $("#classes"); cl.innerHTML = "";
    Object.entries(s.by_class).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
      const sp = document.createElement("span"); sp.innerHTML = `${k.replace(/_/g, " ")}<b>${v}</b>`; cl.appendChild(sp);
    });
  }
  function renderCands(c, chosen) {
    lastCands = c; const ul = $("#cands"); ul.innerHTML = "";
    const top = Math.max(...c.map(x => x.p), 1e-6);
    c.forEach(x => {
      const li = document.createElement("li");
      if (x.uci === chosen) li.classList.add("chosen"); if (x.repeats) li.classList.add("rep");
      li.innerHTML = `<span class="san">${x.san}</span><div class="bar"><i style="width:${(100 * x.p / top).toFixed(1)}%"></i></div><span class="p">${(100 * x.p).toFixed(1)}%</span>`;
      ul.appendChild(li);
    });
    drawArrows(c.slice(0, 3), chosen);
  }
  function sqCenter(sq) {
    const f = sq.charCodeAt(0) - 97, r = parseInt(sq[1]) - 1;
    const flip = board.orientation() === "black";
    const x = (flip ? 7 - f : f) * 12.5 + 6.25, y = (flip ? r : 7 - r) * 12.5 + 6.25;
    return [x, y];
  }
  function drawArrows(c, chosen) {
    const svg = $("#arrows"); svg.innerHTML = "";
    if (!c || !c.length) return;
    c.slice(0, 3).forEach((x, i) => {
      const [x1, y1] = sqCenter(x.uci.slice(0, 2)), [x2, y2] = sqCenter(x.uci.slice(2, 4));
      const main = x.uci === chosen;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", x1); line.setAttribute("y1", y1); line.setAttribute("x2", x2); line.setAttribute("y2", y2);
      line.setAttribute("stroke", main ? "#f5b942" : "#4cc9f0"); line.setAttribute("stroke-width", main ? 1.6 : 0.9);
      line.setAttribute("stroke-linecap", "round"); line.setAttribute("opacity", main ? 0.95 : 0.45 - i * 0.1);
      svg.appendChild(line);
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", x2); dot.setAttribute("cy", y2); dot.setAttribute("r", main ? 2 : 1.3); dot.setAttribute("fill", line.getAttribute("stroke")); dot.setAttribute("opacity", line.getAttribute("opacity"));
      svg.appendChild(dot);
    });
  }

  // ---------------------------------------------------------------- setup
  async function init() {
    const info = await (await fetch("/api/info")).json();
    $("#n-neurons").textContent = info.neurons.toLocaleString();
    $("#enc-info").textContent = `Input: ${info.input_neurons} visual projection neurons. ` +
      Object.entries(info.variants).map(([k, v]) => `${k} readout agrees with Stockfish ${(100 * v).toFixed(1)}% of the time`).join("; ") + ".";
  }
  function newGame() {
    game.reset(); history = [game.fen()]; thinking = false; lastCands = []; scene?.reset(); scene?.setThinking(false);
    board.orientation(humanColor() === "w" ? "white" : "black");
    board.position("start"); highlight(); renderMoves(); drawArrows([]);
    $("#cands").innerHTML = '<li class="empty">Candidate moves appear here after the fly thinks.</li>';
    $("#classes").innerHTML = ""; $("#tl").style.width = "0"; $("#tl-label").textContent = "0 ms";
    ["driven", "neurons", "spikes", "dn", "time"].forEach(k => $("#s-" + k).textContent = "–");
    updateStatus();
    if (humanColor() === "b") setTimeout(flyMove, 300);
  }
  $("#new").addEventListener("click", newGame);
  $("#color").addEventListener("change", newGame);
  init().then(newGame);
})();
