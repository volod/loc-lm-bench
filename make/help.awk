BEGIN {
  command_width = 34
  description_width = 76
  continuation = sprintf("%" (command_width + 6) "s", "")

  print "Usage:"
  print "  make <target> [VAR=value...]"
  print ""
  print "Targets:"
}

/^##@ / {
  section = substr($0, 5)
  if (printed_section) {
    print ""
  }
  printf "  %s:\n", section
  printed_section = 1
  next
}

/^[A-Za-z0-9_.-]+:.*## / {
  target = $0
  sub(/:.*/, "", target)

  description = $0
  sub(/^[^:]+:.*##[ \t]*/, "", description)

  if (!printed_section) {
    print "  Other:"
    printed_section = 1
  }
  print_target(target, description)
}

function print_target(target, description,    prefix, count, words, line, i, word) {
  prefix = sprintf("    %-" command_width "s  ", target)
  count = split(description, words, /[ \t]+/)
  line = ""

  for (i = 1; i <= count; i++) {
    word = words[i]
    if (line == "") {
      line = word
    } else if (length(line) + 1 + length(word) <= description_width) {
      line = line " " word
    } else {
      print_wrapped_line(prefix, line)
      prefix = continuation
      line = word
    }
  }

  print_wrapped_line(prefix, line)
}

function print_wrapped_line(prefix, line) {
  if (line == "") {
    print prefix
  } else {
    print prefix line
  }
}
