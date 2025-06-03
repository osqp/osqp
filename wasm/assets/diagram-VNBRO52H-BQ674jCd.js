import{p as B}from"./chunk-4BMEZGHF-z14ycCT3.js";import{_ as l,s as F,g as P,q as W,p as S,a as z,b as T,E as x,I as v,d as D,y as E,F as A,G as R,l as $}from"./mermaid-Da56-Xo5.js";import{p as Y}from"./radar-MK3ICKWK-BZDr5wsz.js";import"./index-aRpq2G87.js";import"./transform-Cyp0GDF-.js";import"./timer-DFzT7np-.js";import"./step-BwsUM5iJ.js";import"./_baseEach-B07JCD58.js";import"./_baseUniq-CztUjN3r.js";import"./min-Zg4iO1C5.js";import"./_baseMap-CLtXrJAu.js";import"./clone-ih2DSwuN.js";import"./_createAggregator-DGsbTvcV.js";var w={packet:[]},u=structuredClone(w),H=R.packet,I=l(()=>{const t=x({...H,...A().packet});return t.showBits&&(t.paddingY+=10),t},"getConfig"),L=l(()=>u.packet,"getPacket"),m={pushWord:l(t=>{t.length>0&&u.packet.push(t)},"pushWord"),getPacket:L,getConfig:I,clear:l(()=>{E(),u=structuredClone(w)},"clear"),setAccTitle:T,getAccTitle:z,setDiagramTitle:S,getDiagramTitle:W,getAccDescription:P,setAccDescription:F},j=l(t=>{B(t,m);let e=-1,o=[],i=1;const{bitsPerRow:s}=m.getConfig();for(let{start:a,end:r,label:p}of t.blocks){if(r&&r<a)throw new Error(`Packet block ${a} - ${r} is invalid. End must be greater than start.`);if(a!==e+1)throw new Error(`Packet block ${a} - ${r??a} is not contiguous. It should start from ${e+1}.`);for(e=r??a,$.debug(`Packet block ${a} - ${e} with label ${p}`);o.length<=s+1&&m.getPacket().length<1e4;){const[h,c]=q({start:a,end:r,label:p},i,s);if(o.push(h),h.end+1===i*s&&(m.pushWord(o),o=[],i++),!c)break;({start:a,end:r,label:p}=c)}}m.pushWord(o)},"populate"),q=l((t,e,o)=>{if(t.end===void 0&&(t.end=t.start),t.start>t.end)throw new Error(`Block start ${t.start} is greater than block end ${t.end}.`);return t.end+1<=e*o?[t,void 0]:[{start:t.start,end:e*o-1,label:t.label},{start:e*o,end:t.end,label:t.label}]},"getNextFittingBlock"),G={parse:l(async t=>{const e=await Y("packet",t);$.debug(e),j(e)},"parse")},M=l((t,e,o,i)=>{const s=i.db,a=s.getConfig(),{rowHeight:r,paddingY:p,bitWidth:h,bitsPerRow:c}=a,f=s.getPacket(),n=s.getDiagramTitle(),k=r+p,d=k*(f.length+1)-(n?0:r),b=h*c+2,g=v(e);g.attr("viewbox",`0 0 ${b} ${d}`),D(g,d,b,a.useMaxWidth);for(const[y,C]of f.entries())N(g,C,y,a);g.append("text").text(n).attr("x",b/2).attr("y",d-k/2).attr("dominant-baseline","middle").attr("text-anchor","middle").attr("class","packetTitle")},"draw"),N=l((t,e,o,{rowHeight:i,paddingX:s,paddingY:a,bitWidth:r,bitsPerRow:p,showBits:h})=>{const c=t.append("g"),f=o*(i+a)+a;for(const n of e){const k=n.start%p*r+1,d=(n.end-n.start+1)*r-s;if(c.append("rect").attr("x",k).attr("y",f).attr("width",d).attr("height",i).attr("class","packetBlock"),c.append("text").attr("x",k+d/2).attr("y",f+i/2).attr("class","packetLabel").attr("dominant-baseline","middle").attr("text-anchor","middle").text(n.label),!h)continue;const b=n.end===n.start,g=f-2;c.append("text").attr("x",k+(b?d/2:0)).attr("y",g).attr("class","packetByte start").attr("dominant-baseline","auto").attr("text-anchor",b?"middle":"start").text(n.start),b||c.append("text").attr("x",k+d).attr("y",g).attr("class","packetByte end").attr("dominant-baseline","auto").attr("text-anchor","end").text(n.end)}},"drawWord"),X={byteFontSize:"10px",startByteColor:"black",endByteColor:"black",labelColor:"black",labelFontSize:"12px",titleColor:"black",titleFontSize:"14px",blockStrokeColor:"black",blockStrokeWidth:"1",blockFillColor:"#efefef"},_={parser:G,db:m,renderer:{draw:M},styles:l(({packet:t}={})=>{const e=x(X,t);return`
	.packetByte {
		font-size: ${e.byteFontSize};
	}
	.packetByte.start {
		fill: ${e.startByteColor};
	}
	.packetByte.end {
		fill: ${e.endByteColor};
	}
	.packetLabel {
		fill: ${e.labelColor};
		font-size: ${e.labelFontSize};
	}
	.packetTitle {
		fill: ${e.titleColor};
		font-size: ${e.titleFontSize};
	}
	.packetBlock {
		stroke: ${e.blockStrokeColor};
		stroke-width: ${e.blockStrokeWidth};
		fill: ${e.blockFillColor};
	}
	`},"styles")};export{_ as diagram};
