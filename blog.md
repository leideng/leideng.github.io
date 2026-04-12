---
title: Blog
description: Writing by Lei Deng on efficient AI, sparse attention, and large model systems.
kicker: Blog
header_title: Blog
lead: Notes and writing on efficient AI, sparse attention, long-context inference, and engineering for large model systems.
---

{% for post in site.posts %}
## [{{ post.title }}]({{ post.url | relative_url }})

{{ post.date | date: "%B %-d, %Y" }}

{% if post.excerpt %}{{ post.excerpt | strip_html | strip | truncate: 220 }}{% endif %}

{% endfor %}
