#!/usr/bin/python3

with open("test.txt", "r") as f:
  txt = f.read()
  split_txt = txt.split("Document")
  count = 0
  for elem in split_txt:
    count+=1
  print(count)
