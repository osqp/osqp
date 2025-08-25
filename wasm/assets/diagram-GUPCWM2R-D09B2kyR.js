var m;import{p as C}from"./chunk-ANTBXLJU-ZgvmG8Pb.js";import{_ as f,E as u,I as v,d as P,l as w,b as z,a as F,p as S,q as E,g as W,s as T,F as D,G as A,y as R}from"./mermaid-D59lkToe.js";import{p as Y}from"./treemap-75Q7IDZK-Bh5pVSmo.js";import"./index-D4bXoNM3.js";import"./transform-B8bpuzxV.js";import"./timer-BwIYMJWC.js";import"./step-BwsUM5iJ.js";import"./_baseEach-CIMlsWNn.js";import"./_baseUniq-CGK6su7v.js";import"./min-DrLfF3uL.js";import"./_baseMap-1GEe_WcR.js";import"./clone-DFaYgbfI.js";import"./_createAggregator--z161kAx.js";var j=A.packet,y=(m=class{constructor(){this.packet=[],this.setAccTitle=z,this.getAccTitle=F,this.setDiagramTitle=S,this.getDiagramTitle=E,this.getAccDescription=W,this.setAccDescription=T}getConfig(){const t=u({...j,...D().packet});return t.showBits&&(t.paddingY+=10),t}getPacket(){return this.packet}pushWord(t){t.length>0&&this.packet.push(t)}clear(){R(),this.packet=[]}},f(m,"PacketDB"),m),H=f((e,t)=>{C(e,t);let o=-1,r=[],n=1;const{bitsPerRow:l}=t.getConfig();for(let{start:a,end:i,bits:d,label:c}of e.blocks){if(a!==void 0&&i!==void 0&&i<a)throw new Error(`Packet block ${a} - ${i} is invalid. End must be greater than start.`);if(a??(a=o+1),a!==o+1)throw new Error(`Packet block ${a} - ${i??a} is not contiguous. It should start from ${o+1}.`);if(d===0)throw new Error(`Packet block ${a} is invalid. Cannot have a zero bit field.`);for(i??(i=a+(d??1)-1),d??(d=i-a+1),o=i,w.debug(`Packet block ${a} - ${o} with label ${c}`);r.length<=l+1&&t.getPacket().length<1e4;){const[p,s]=I({start:a,end:i,bits:d,label:c},n,l);if(r.push(p),p.end+1===n*l&&(t.pushWord(r),r=[],n++),!s)break;({start:a,end:i,bits:d,label:c}=s)}}t.pushWord(r)},"populate"),I=f((e,t,o)=>{if(e.start===void 0)throw new Error("start should have been set during first phase");if(e.end===void 0)throw new Error("end should have been set during first phase");if(e.start>e.end)throw new Error(`Block start ${e.start} is greater than block end ${e.end}.`);if(e.end+1<=t*o)return[e,void 0];const r=t*o-1,n=t*o;return[{start:e.start,end:r,label:e.label,bits:r-e.start},{start:n,end:e.end,label:e.label,bits:e.end-n}]},"getNextFittingBlock"),x={parser:{yy:void 0},parse:f(async e=>{var r;const t=await Y("packet",e),o=(r=x.parser)==null?void 0:r.yy;if(!(o instanceof y))throw new Error("parser.parser?.yy was not a PacketDB. This is due to a bug within Mermaid, please report this issue at https://github.com/mermaid-js/mermaid/issues.");w.debug(t),H(t,o)},"parse")},L=f((e,t,o,r)=>{const n=r.db,l=n.getConfig(),{rowHeight:a,paddingY:i,bitWidth:d,bitsPerRow:c}=l,p=n.getPacket(),s=n.getDiagramTitle(),h=a+i,b=h*(p.length+1)-(s?0:a),k=d*c+2,g=v(t);g.attr("viewbox",`0 0 ${k} ${b}`),P(g,b,k,l.useMaxWidth);for(const[$,B]of p.entries())M(g,B,$,l);g.append("text").text(s).attr("x",k/2).attr("y",b-h/2).attr("dominant-baseline","middle").attr("text-anchor","middle").attr("class","packetTitle")},"draw"),M=f((e,t,o,{rowHeight:r,paddingX:n,paddingY:l,bitWidth:a,bitsPerRow:i,showBits:d})=>{const c=e.append("g"),p=o*(r+l)+l;for(const s of t){const h=s.start%i*a+1,b=(s.end-s.start+1)*a-n;if(c.append("rect").attr("x",h).attr("y",p).attr("width",b).attr("height",r).attr("class","packetBlock"),c.append("text").attr("x",h+b/2).attr("y",p+r/2).attr("class","packetLabel").attr("dominant-baseline","middle").attr("text-anchor","middle").text(s.label),!d)continue;const k=s.end===s.start,g=p-2;c.append("text").attr("x",h+(k?b/2:0)).attr("y",g).attr("class","packetByte start").attr("dominant-baseline","auto").attr("text-anchor",k?"middle":"start").text(s.start),k||c.append("text").attr("x",h+b).attr("y",g).attr("class","packetByte end").attr("dominant-baseline","auto").attr("text-anchor","end").text(s.end)}},"drawWord"),q={draw:L},G={byteFontSize:"10px",startByteColor:"black",endByteColor:"black",labelColor:"black",labelFontSize:"12px",titleColor:"black",titleFontSize:"14px",blockStrokeColor:"black",blockStrokeWidth:"1",blockFillColor:"#efefef"},N=f(({packet:e}={})=>{const t=u(G,e);return`
	.packetByte {
		font-size: ${t.byteFontSize};
	}
	.packetByte.start {
		fill: ${t.startByteColor};
	}
	.packetByte.end {
		fill: ${t.endByteColor};
	}
	.packetLabel {
		fill: ${t.labelColor};
		font-size: ${t.labelFontSize};
	}
	.packetTitle {
		fill: ${t.titleColor};
		font-size: ${t.titleFontSize};
	}
	.packetBlock {
		stroke: ${t.blockStrokeColor};
		stroke-width: ${t.blockStrokeWidth};
		fill: ${t.blockFillColor};
	}
	`},"styles"),X={parser:x,get db(){return new y},renderer:q,styles:N};export{X as diagram};
