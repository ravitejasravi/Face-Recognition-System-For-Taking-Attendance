import wikipedia

wiki = wikipedia.page("tumkur")
text = wiki.content
# print(text)

lines = text.split(".")
# print(lines)

for i in range(len(lines)):
    lines[i] = " > " + lines[i]

text = '\n'.join(lines)
print(text)
