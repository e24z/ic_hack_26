# magi

- base agent has two functionalities, to search and or read data and summarise it. everything is built on the parallelisation of this phenomnena
- when a heuristic is validated, data is summarised. summary is passed through vllm's halugate to validate it. 
- few layers, initial research, at this stage data is gleaned from the internet.
- to go to the next layer, data is summarised and validated by halugate. % tokens grounded in the facts.
- should be some metric for when to parallelise.
- after the research phase, focus should be synthesis, i.e., mass summarisation and validation.
- evidently, this is just the mass repetition of relatively simple operations.
- should probably also be visualisable in a simple web view


Base object:
- search()
- summarise()
- split()

overseer:
- validate()

- at the end of the research phase, it's just a massive summarise() & valiate() call no?
- if so, is it feasible to have many instances of one agent continually judging itself
