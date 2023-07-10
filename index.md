---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: default
---

# Posts

{% for post in site.posts %}
**{{ post.date | date: "%m-%d-%Y" }}** [{{ post.title }}]({{ post.url }})
{% endfor %}
