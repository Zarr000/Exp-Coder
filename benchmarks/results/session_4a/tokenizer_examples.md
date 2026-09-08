# Session 4A tokenizer qualitative audit

## markdown (0000_markdown.md)
- source chars: 600, tokens: 163, roundtrip: PASS
```text
pieces: # | ĠExp | - | Coder | Ċ | Ċ | ** | O | riginal | Ġlanguage | Ġmodel | Ġfrom | ĠExp | ĠWorks | . | ĠA | uthor | : | ĠZarr | . | ** | Ċ | Ċ | Exp | - | Coder | Ġis | Ġan | Ġoriginal | , | Ġdecoder | - | only | ĠGPT | - | style | Ġtransformer | Ġfor | Ċ | soft | ware | Ġdevelopment | , | Ġbuilt | Ġfrom | Ġscratch | Ġin | ĠPython | Ġ+ | ĠPyTorch | . | Ċ | Ċ | Current | Ġdevelopment | Ġmodel | : | Ġ** | Exp | - | Coder | Ġ120 | M | ** | Ċ | ( | ` | configs | / | exp | _ | coder | _ | 120 | m | . | yaml | `): | Ċ | Ċ
```

## python (0002_python.py)
- source chars: 600, tokens: 146, roundtrip: PASS
```text
pieces: """ | Shared | Ġhelpers | Ġfor | ĠExp | - | Coder | Ġscientific | Ġbenchmarks | . | Ċ | Ċ | K | e | pt | Ġintentionally | Ġsmall | : | Ġseeds | , | Ġresult | Ġserialization | Ġ( | JSON | Ġ+ | Ġmetadata | ), | Ċ | runtime | Ġmemory | Ġprob | ing | Ġ( | W | indow | s | Ġctypes | Ġfallback | Ġwhen | Ġpsutil | Ġis | Ġabsent | ), | Ġand | Ċ | CUDA | Ġdiagnostics | . | Ċ | """ | Ċ | Ċ | import | Ġjson | Ċ | import | Ġos | Ċ | from | Ġdatetime | Ġimport | Ġdatetime | , | Ġtime | zone | Ċ | from | Ġpathlib | Ġimport | ĠPath | Ċ | from | Ġtyping | Ġimport | ĠAny | , | ĠDict | , | ĠOptional | Ċ
```

## cpp (0003_cpp.cpp)
- source chars: 600, tokens: 189, roundtrip: PASS
```text
pieces: // | ĠC | ++ | Ġpatterns | : | Ġtemplates | , | ĠSTL | , | Ġclasses | , | Ġsmart | Ġpointers | , | Ġlambdas | . | Ċ | Ċ | # | include | Ġ< | algorithm | > | Ċ | Ċ | # | include | Ġ< | memory | > | Ċ | Ċ | # | include | Ġ< | string | > | Ċ | Ċ | # | include | Ġ< | unordered | _ | map | > | Ċ | Ċ | # | include | Ġ< | vector | > | ĊĊĊ | Ċ | template | Ġ< | typename | ĠT | > | Ċ | Ċ | class | ĠStack | Ġ{ | ĊĊ | Ġpublic | : | ĊĊĠ | Ġexplicit | ĠStack | ( | std | :: | string | Ġname | ) | Ġ: | Ġname | _(
```

## go (0006_go.go)
- source chars: 600, tokens: 194, roundtrip: PASS
```text
pieces: // | ĠGo | Ġpatterns | : | Ġgoroutines | , | Ġstructs | / | interfaces | , | Ġslices | , | Ġerror | Ġhandling | . | Ċ | Ċ | package | Ġmain | ĊĊĊ | Ċ | import | Ġ( | ĊĊĠĠĠ | Ġ" | errors | " | ĊĊĠĠĠ | Ġ" | fmt | " | ĊĊĠĠĠ | Ġ" | sync | " | Ċ | Ċ | ) | ĊĊĊ | Ċ | const | ĠmaxRetries | Ġ= | Ġ3 | ĊĊĊ | Ċ | func | Ġfibonacci | ( | n | Ġint | ) | Ġint | Ġ{ | ĊĊĠĠĠ | Ġif | Ġn | Ġ<= | Ġ1 | Ġ{ | ĊĊĠĠĠĠĠĠĠ | Ġreturn | Ġn | ĊĊĠĠĠ | Ġ} | ĊĊĠĠĠ | Ġreturn | Ġfibonacci | ( | n | - | 1 | ) | Ġ+ | Ġfibonacci | ( | n | - | 2 | )
```

## javascript (0009_javascript.js)
- source chars: 600, tokens: 197, roundtrip: PASS
```text
pieces: // | ĠDemonstrate | Ġcommon | ĠJavaScript | / | TypeScript | Ġpatterns | . | Ċ | Ċ | const | ĠMAX | _ | RETRIES | Ġ= | Ġ3 | ; | Ċ | Ċ | const | Ġratio | Ġ= | Ġ0 | . | 6180339887498949 | ; | ĊĊĊ | Ċ | function | Ġfibonacci | ( | n | ) | Ġ{ | ĊĊĠ | Ġif | Ġ( | n | Ġ<= | Ġ1 | ) | Ġreturn | Ġn | ; | ĊĊĠ | Ġreturn | Ġfibonacci | ( | n | Ġ- | Ġ1 | ) | Ġ+ | Ġfibonacci | ( | n | Ġ- | Ġ2 | ); | Ċ | Ċ | } | ĊĊĊ | Ċ | function | Ġprimes | ( | limit | ) | Ġ{ | ĊĊĠ | Ġconst | Ġsieve | Ġ= | Ġnew | ĠArray | ( | limit | Ġ+ | Ġ1
```

## json (0012_json.json)
- source chars: 487, tokens: 166, roundtrip: PASS
```text
pieces: { | ĊĊĠ | Ġ" | name | ": | Ġ" | exp | - | coder | ", | ĊĊĠ | Ġ" | version | ": | Ġ" | 0 | . | 1 | . | 0 | ", | ĊĊĠ | Ġ" | debug | ": | Ġtrue | , | ĊĊĠ | Ġ" | max | _ | retries | ": | Ġ3 | , | ĊĊĠ | Ġ" | ratio | ": | Ġ0 | . | 6180339887498949 | , | ĊĊĠ | Ġ" | tags | ": | Ġ[" | model | ", | Ġ" | tokenizer | ", | Ġ" | benchmark | "], | ĊĊĠ | Ġ" | model | ": | Ġ{ | ĊĊĠĠĠ | Ġ" | vocab | _ | size | ": | Ġ50304 | , | ĊĊĠĠĠ | Ġ" | hidden | _ | size | ": | Ġ768 | , | ĊĊĠĠĠ | Ġ" | num
```

## rust (0017_rust.rs)
- source chars: 600, tokens: 188, roundtrip: PASS
```text
pieces: // | ĠRust | Ġpatterns | : | Ġownership | , | ĠResult | , | Ġstructs | , | Ġiterators | , | Ġmatch | . | Ċ | Ċ | use | Ġstd | :: | collections | :: | HashMap | ; | Ċ | Ċ | use | Ġstd | :: | error | :: | Error | ; | ĊĊĊ | Ċ | const | ĠMAX | _ | RETRIES | : | Ġu | 32 | Ġ= | Ġ3 | ; | ĊĊĊ | Ċ | fn | Ġfibonacci | ( | n | : | Ġu | 32 | ) | Ġ-> | Ġu | 32 | Ġ{ | ĊĊĠĠĠ | Ġmatch | Ġn | Ġ{ | ĊĊĠĠĠĠĠĠĠ | Ġ0 | Ġ| | Ġ1 | Ġ=> | Ġn | , | ĊĊĠĠĠĠĠĠĠ | Ġ_ | Ġ=> | Ġfibonacci | ( | n | Ġ- | Ġ1 | ) | Ġ+ | Ġfibonacci
```

## shell (0020_shell.sh)
- source chars: 600, tokens: 205, roundtrip: PASS
```text
pieces: #!/ | usr | / | bin | / | env | Ġbash | Ċ | Ċ | # | ĠDemonstrates | Ġshell | Ġconstructs | Ġfor | Ġthe | Ġtokenizer | Ġcorpus | . | Ċ | Ċ | set | Ġ- | euo | Ġpipefail | ĊĊĊ | Ċ | MAX | _ | RETRIES | = | 3 | Ċ | Ċ | ratio | = | 0 | . | 6180339887498949 | ĊĊĊ | Ċ | log | () | Ġ{ | ĊĊĠ | Ġlocal | Ġlevel | ="$ | 1 | " | ĊĊĠ | Ġshift | ĊĊĠ | Ġecho | Ġ"[${ | level | }] | Ġ$*" | Ċ | Ċ | } | ĊĊĊ | Ċ | fib | () | Ġ{ | ĊĊĠ | Ġlocal | Ġn | =$ | 1 | ĊĊĠ | Ġif | Ġ(( | Ġn | Ġ<= | Ġ1 | Ġ)); | Ġthen | ĊĊĠĠĠ | Ġecho
```

## yaml (0021_yaml.yaml)
- source chars: 416, tokens: 132, roundtrip: PASS
```text
pieces: # | ĠDeployment | Ġconfiguration | . | Ċ | Ċ | app | : | Ġexp | - | coder | Ċ | Ċ | port | : | Ġ8080 | Ċ | Ċ | replicas | : | Ġ2 | Ċ | Ċ | features | : | ĊĊĠ | Ġcode | _ | generation | : | Ġtrue | ĊĊĠ | Ġmultimodal | : | Ġfalse | Ċ | Ċ | servers | : | ĊĊĠ | Ġ- | Ġhost | : | Ġ10 | . | 0 | . | 0 | . | 1 | ĊĊĠĠĠ | Ġrole | : | Ġprimary | ĊĊĠ | Ġ- | Ġhost | : | Ġ10 | . | 0 | . | 0 | . | 2 | ĊĊĠĠĠ | Ġrole | : | Ġreplica | Ċ | Ċ | training | : | ĊĊĠ | Ġlearning | _ | rate | : | Ġ3 | .
```

