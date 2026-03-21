---
title: Home
description: Lei Deng personal website
---

<section class="hero">
  <div class="hero-photo">
    <img src="{{ '/photo.jpeg' | relative_url }}" alt="Portrait of Lei Deng">
  </div>
  <div class="hero-copy">
    <p class="section-kicker">Personal Website</p>
    <h1>Lei Deng</h1>
    <p class="lead">Researcher working across efficient AI, sparse attention, and communication systems.</p>
    <p>
      The latest public profile attached to this site is the Google Scholar record for
      <a href="https://scholar.google.com/citations?user=Bx2v5H8AAAAJ&hl=en">Lei Deng</a>,
      which currently lists <strong>Hong Kong Theory Lab, Huawei</strong> and research interests including
      efficient AI and sparse attention. Earlier education and academic background from the previous site are preserved below.
    </p>
    <div class="button-row">
      <a class="button-link" href="{{ '/publications/' | relative_url }}">View Publications</a>
      <a class="button-link secondary" href="{{ '/blog/' | relative_url }}">Read Blog</a>
      <a class="button-link secondary" href="https://github.com/leideng">GitHub</a>
      <a class="button-link secondary" href="https://scholar.google.com/citations?user=Bx2v5H8AAAAJ&hl=en">Google Scholar</a>
    </div>
  </div>
</section>

<section class="grid-two">
  <div class="panel">
    <h2>Biography</h2>
    <p>
      Lei Deng received a BEng from Shanghai Jiao Tong University in 2012 and a PhD from
      The Chinese University of Hong Kong in 2017. Previous versions of this site listed an
      assistant professorship at Shenzhen University and a postdoctoral appointment at CUHK.
    </p>
    <p>
      The recent publication record indicates an active research focus spanning large language models,
      multimodal learning, diffusion-based compression, and communication systems.
    </p>
  </div>
  <div class="panel">
    <h2>Research Areas</h2>
    <ul class="stack-list">
      <li>Efficient AI and sparse attention</li>
      <li>Long-context language modeling</li>
      <li>Multimodal adaptation and reasoning</li>
      <li>Wireless and delay-constrained networking</li>
      <li>Distributed storage and systems optimization</li>
    </ul>
  </div>
</section>

<section class="grid-two">
  <div class="panel">
    <h2>Contact</h2>
    <ul class="meta-list">
      <li>Email: <a href="mailto:ldeng.sjtu@gmail.com">ldeng.sjtu@gmail.com</a></li>
      <li>Google Scholar: <a href="https://scholar.google.com/citations?user=Bx2v5H8AAAAJ&hl=en">profile</a></li>
      <li>GitHub: <a href="https://github.com/leideng">@leideng</a></li>
      <li>Website: <a href="https://leideng.github.io/">leideng.github.io</a></li>
    </ul>
  </div>
  <div class="panel">
    <h2>Recent Publications</h2>
    <ul class="plain-list">
      {% for publication in site.data.publications limit: 5 %}
      <li>
        <strong>{{ publication.title }}</strong><br>
        <span class="pub-meta">{{ publication.venue }}, {{ publication.year }}</span>
      </li>
      {% endfor %}
    </ul>
    <p><a href="{{ '/publications/' | relative_url }}">See the full recent list</a></p>
  </div>
</section>

<section class="grid-two">
  <div class="panel">
    <h2>Notes</h2>
    <p>
      This site has been rebuilt as a Jekyll site where the content pages and blog posts are plain Markdown files.
      It is intended to stay lightweight, editable, and easy to maintain directly on GitHub.
    </p>
  </div>
  <div class="panel">
    <h2>Latest Blog Posts</h2>
    <ul class="post-list">
      {% for post in site.posts limit: 3 %}
      <li>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a><br>
        <span class="pub-meta">{{ post.date | date: "%B %-d, %Y" }}</span>
      </li>
      {% endfor %}
    </ul>
  </div>
</section>
