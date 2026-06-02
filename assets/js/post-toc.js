(function () {
  var content = document.querySelector(".post-content");
  var nav = document.getElementById("post-toc-nav");
  if (!content || !nav) {
    return;
  }

  var headings = content.querySelectorAll("h2, h3");
  if (!headings.length) {
    return;
  }

  var list = document.createElement("ul");
  list.className = "post-toc-list";
  var currentH2Item = null;
  var currentSubList = null;
  var links = [];

  headings.forEach(function (heading) {
    if (!heading.id) {
      heading.id = heading.textContent
        .trim()
        .toLowerCase()
        .replace(/[^\w\s-]/g, "")
        .replace(/\s+/g, "-");
    }

    var item = document.createElement("li");
    var link = document.createElement("a");
    link.href = "#" + heading.id;
    link.textContent = heading.textContent.replace(/\s+/g, " ").trim();
    link.className = heading.tagName === "H2" ? "post-toc-link post-toc-h2" : "post-toc-link post-toc-h3";
    link.dataset.target = heading.id;
    item.appendChild(link);
    links.push({ link: link, heading: heading });

    if (heading.tagName === "H2") {
      currentH2Item = item;
      currentSubList = null;
      list.appendChild(item);
      return;
    }

    if (!currentSubList && currentH2Item) {
      currentSubList = document.createElement("ul");
      currentSubList.className = "post-toc-sublist";
      currentH2Item.appendChild(currentSubList);
    }

    (currentSubList || list).appendChild(item);
  });

  nav.appendChild(list);

  links.forEach(function (entry) {
    entry.link.addEventListener("click", function (event) {
      event.preventDefault();
      entry.heading.scrollIntoView({ behavior: "smooth", block: "start" });
      history.replaceState(null, "", "#" + entry.heading.id);
      setActive(entry.link);
    });
  });

  function setActive(activeLink) {
    links.forEach(function (entry) {
      entry.link.classList.toggle("is-active", entry.link === activeLink);
    });
  }

  function updateActiveLink() {
    var marker = window.scrollY + 120;
    var active = links[0].link;

    links.forEach(function (entry) {
      if (entry.heading.offsetTop <= marker) {
        active = entry.link;
      }
    });

    setActive(active);
  }

  window.addEventListener("scroll", updateActiveLink, { passive: true });
  window.addEventListener("resize", updateActiveLink);
  updateActiveLink();
})();
