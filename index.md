---
title: Home
description: Lei Deng personal website
kicker: Personal Website
hero_title: Lei Deng
lead: Researcher working across efficient AI, sparse attention, and communication systems.
hero_image: /photo.jpeg
hero_image_alt: Portrait of Lei Deng
hero_intro: The latest public profile attached to this site is the Google Scholar record for Lei Deng, which currently lists Hong Kong Theory Lab, Huawei and research interests including efficient AI and sparse attention. Earlier education and academic background from the previous site are preserved below.
hero_buttons:
  - label: View Publications
    url: /publications/
  - label: Read Blogs
    url: /blog/
    style: secondary
  - label: GitHub
    url: https://github.com/leideng
    style: secondary
  - label: Google Scholar
    url: https://scholar.google.com/citations?user=Bx2v5H8AAAAJ&hl=en
    style: secondary
---

## Biography

Lei Deng received a BEng from Shanghai Jiao Tong University in 2012 and a PhD from The Chinese University of Hong Kong in 2017. Previous versions of this site listed an assistant professorship at Shenzhen University and a postdoctoral appointment at CUHK.

The recent publication record indicates an active research focus spanning large language models, multimodal learning, diffusion-based compression, and communication systems.

## Research Areas

- Efficient AI and sparse attention
- Long-context language modeling
- Multimodal adaptation and reasoning
- Wireless and delay-constrained networking
- Distributed storage and systems optimization

## Contact

- Email: [ldeng.sjtu@gmail.com](mailto:ldeng.sjtu@gmail.com)
- Google Scholar: [profile](https://scholar.google.com/citations?user=Bx2v5H8AAAAJ&hl=en)
- GitHub: [@leideng](https://github.com/leideng)
- Website: [leideng.github.io](https://leideng.github.io/)

## Recent Publications

{% for publication in site.data.publications limit: 5 %}
- **{{ publication.title }}**
  {{ publication.venue }}, {{ publication.year }}
{% endfor %}

[See the full recent list]({{ '/publications/' | relative_url }})

## Notes

This site has been rebuilt as a Jekyll site where the content pages and blog posts are plain Markdown files. It is intended to stay lightweight, editable, and easy to maintain directly on GitHub.

## Latest Blog Posts

{% for post in site.posts limit: 3 %}
- [{{ post.title }}]({{ post.url | relative_url }})
  {{ post.date | date: "%B %-d, %Y" }}
{% endfor %}

## More

- [Teaching]({{ '/teaching/' | relative_url }})
- [Talks]({{ '/talks/' | relative_url }})
- [Awards]({{ '/awards/' | relative_url }})
- [Services]({{ '/services/' | relative_url }})
