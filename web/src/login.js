let shown = false;

// Full-page login over whatever the page was showing. Reloads on success.
export function showLogin() {
  if (shown) return;
  shown = true;
  const wrap = document.createElement("div");
  wrap.className = "login";
  wrap.innerHTML = `
    <form class="login-box" autocomplete="off">
      <h1>radar-desk</h1>
      <p class="muted">Research use only. Enter the owner token.</p>
      <label>Token <input type="password" name="token" required autofocus></label>
      <button type="submit">Log in</button>
      <p class="error" role="alert" hidden></p>
    </form>`;
  document.body.append(wrap);
  const form = wrap.querySelector("form");
  const error = wrap.querySelector(".error");
  form.token.focus();
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    error.hidden = true;
    const res = await fetch("/auth/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: form.token.value }),
    });
    if (res.ok) {
      location.reload();
      return;
    }
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    error.textContent = detail;
    error.hidden = false;
  });
}

export function wireLogout() {
  const button = document.getElementById("logout");
  if (!button) return;
  button.addEventListener("click", async () => {
    await fetch("/auth/logout", { method: "POST", credentials: "same-origin" });
    location.reload();
  });
}
