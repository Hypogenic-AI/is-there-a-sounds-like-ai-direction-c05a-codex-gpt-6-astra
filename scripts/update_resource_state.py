from pathlib import Path
import sys
p=Path('STATE.md')
s=p.read_text(); start='<!-- NEURICO_AGENT_NOTES_START:resource_finder -->'; end='<!-- NEURICO_AGENT_NOTES_END:resource_finder -->'
a,b=s.split(start,1); _,c=b.split(end,1)
p.write_text(a+start+'\n'+sys.stdin.read().strip()+'\n'+end+c)
