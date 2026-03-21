document.addEventListener("DOMContentLoaded", function () {
  var button = document.getElementById("theme-toggle");

  if (!button) {
    return;
  }

  button.addEventListener("click", function () {
    var currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    var nextTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("theme", nextTheme);
  });
});
