document.addEventListener("DOMContentLoaded", function () {
  var button = document.getElementById("theme-toggle");
  var mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

  if (!button) {
    return;
  }

  function applyMode(mode) {
    var theme = mode === "auto" ? (mediaQuery.matches ? "dark" : "light") : mode;
    document.documentElement.setAttribute("data-mode", mode);
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("mode", mode);
    syncButtonLabel(mode);
  }

  function syncButtonLabel(mode) {
    var labels = {
      light: "Color mode: light",
      dark: "Color mode: dark",
      auto: "Color mode: system"
    };
    var label = labels[mode] || labels.auto;
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
  }

  function nextMode(mode) {
    if (mode === "auto") {
      return "light";
    }
    if (mode === "light") {
      return "dark";
    }
    return "auto";
  }

  syncButtonLabel(document.documentElement.getAttribute("data-mode") || "auto");

  button.addEventListener("click", function () {
    var currentMode = document.documentElement.getAttribute("data-mode") || "auto";
    applyMode(nextMode(currentMode));
  });

  mediaQuery.addEventListener("change", function () {
    var mode = document.documentElement.getAttribute("data-mode") || "auto";
    if (mode === "auto") {
      applyMode("auto");
    }
  });
});
