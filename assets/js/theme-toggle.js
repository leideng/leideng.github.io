document.addEventListener("DOMContentLoaded", function () {
  var button = document.getElementById("theme-toggle");

  if (!button) {
    return;
  }

  function syncButtonLabel() {
    var theme = document.documentElement.getAttribute("data-theme") || "light";
    var nextTheme = theme === "dark" ? "light" : "dark";
    button.setAttribute("aria-label", "Switch to " + nextTheme + " mode");
    button.setAttribute("title", "Switch to " + nextTheme + " mode");
  }

  syncButtonLabel();

  button.addEventListener("click", function () {
    var currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    var nextTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("theme", nextTheme);
    syncButtonLabel();
  });
});
