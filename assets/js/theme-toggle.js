document.addEventListener("DOMContentLoaded", function () {
  var button = document.getElementById("theme-toggle");

  if (!button) {
    return;
  }

  function syncButtonLabel(theme) {
    var nextTheme = theme === "dark" ? "light" : "dark";
    var label = nextTheme === "dark" ? "Switch to dark mode" : "Switch to light mode";
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    syncButtonLabel(theme);
  }

  var currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(currentTheme);

  button.addEventListener("click", function () {
    var theme = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(theme);
  });
});
