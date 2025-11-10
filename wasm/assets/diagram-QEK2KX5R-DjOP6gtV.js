import"./_Uint8Array-D5Z9rM2X.js";import"./isSymbol-xTSywenE.js";import"./_arrayMap-DQI2GUNb.js";import"./toString-CBnct2wx.js";import"./toNumber-CTOPJrpu.js";import"./toInteger-Bor4StOs.js";import"./isArrayLikeObject-DKyJYtr8.js";import"./_getTag-CVRrQL_Q.js";import"./_baseUniq-DHpyQ0iZ.js";import"./reduce-BDKuIPQC.js";import"./_baseIsEqual-CVFfFJm2.js";import"./chunk-4KMFLZZN-ezSWKPWV.js";import"./_toKey-16owuO_q.js";import"./memoize-DhF8x4Aa.js";import"./get-BUGWIXZE.js";import"./_baseFlatten-ViL-mfkw.js";import"./_basePickBy-CSDa7ReR.js";import"./merge-Dvc5opZF.js";import"./_baseSlice-BWZ0PeaJ.js";import"./_arrayReduce-DV4IVFuG.js";import"./clone-Cq89YAb4.js";import"./_baseEach-CMG9RfnR.js";import"./hasIn-BgJKauE7.js";import"./_baseProperty-c4IQJQis.js";import"./_createAggregator-CP8XSJVt.js";import"./min-OzDpO0MS.js";import"./_baseMap-Cw6xPrGQ.js";import"./isString-Cwbr2gIw.js";import"./isEmpty-B-9PYDiJ.js";import"./_baseSet-L2qemYhz.js";import"./uniq-pTjN93Sd.js";import"./preload-helper-BnutJmlU.js";import"./main-ChjX94eX.js";import"./purify.es-CPMOFAIu.js";import"./timer-CTpu0Fa6.js";import"./src-BwH0YUPh.js";import"./math-B-uM5inP.js";import"./step-BDtirdXu.js";import{i as y}from"./chunk-S3R3BYOJ-BW8g0Zog.js";import{n as l,r as k}from"./src-BuVGraFB.js";import{B as O,C as S,T as I,U as z,_ as E,a as F,d as P,v as R,y as w,z as D}from"./chunk-ABZYJK2D-Cm5Mq_1V.js";import{t as B}from"./chunk-EXTU4WIE-DUVoiM8z.js";import"./dist-DiomJpDg.js";import"./chunk-JEIROHC2-DJ3tpwJA.js";import"./chunk-BN7GFLIU-B2OO_dsW.js";import"./chunk-T44TD3VJ-ob7Tm16D.js";import"./chunk-KMC2YHZD-aZ_clHHw.js";import"./chunk-WFWHJNB7-BdOad_I6.js";import"./chunk-WFRQ32O7-DaR0-qal.js";import"./chunk-XRWGC2XP-bK3mWGs8.js";import{t as G}from"./chunk-4BX2VUAB-CEk5Yddb.js";import{t as V}from"./mermaid-parser.core-NXSENIcQ.js";var h={showLegend:!0,ticks:5,max:null,min:0,graticule:"circle"},b={axes:[],curves:[],options:h},g=structuredClone(b),_=P.radar,j=l(()=>y({..._,...w().radar}),"getConfig"),C=l(()=>g.axes,"getAxes"),W=l(()=>g.curves,"getCurves"),H=l(()=>g.options,"getOptions"),N=l(e=>{g.axes=e.map(t=>({name:t.name,label:t.label??t.name}))},"setAxes"),U=l(e=>{g.curves=e.map(t=>({name:t.name,label:t.label??t.name,entries:Z(t.entries)}))},"setCurves"),Z=l(e=>{if(e[0].axis==null)return e.map(r=>r.value);let t=C();if(t.length===0)throw Error("Axes must be populated before curves for reference entries");return t.map(r=>{let a=e.find(i=>{var s;return((s=i.axis)==null?void 0:s.$refText)===r.name});if(a===void 0)throw Error("Missing entry for axis "+r.label);return a.value})},"computeCurveEntries"),x={getAxes:C,getCurves:W,getOptions:H,setAxes:N,setCurves:U,setOptions:l(e=>{var r,a,i,s,n;let t=e.reduce((o,p)=>(o[p.name]=p,o),{});g.options={showLegend:((r=t.showLegend)==null?void 0:r.value)??h.showLegend,ticks:((a=t.ticks)==null?void 0:a.value)??h.ticks,max:((i=t.max)==null?void 0:i.value)??h.max,min:((s=t.min)==null?void 0:s.value)??h.min,graticule:((n=t.graticule)==null?void 0:n.value)??h.graticule}},"setOptions"),getConfig:j,clear:l(()=>{F(),g=structuredClone(b)},"clear"),setAccTitle:O,getAccTitle:R,setDiagramTitle:z,getDiagramTitle:S,getAccDescription:E,setAccDescription:D},q=l(e=>{G(e,x);let{axes:t,curves:r,options:a}=e;x.setAxes(t),x.setCurves(r),x.setOptions(a)},"populate"),J={parse:l(async e=>{let t=await V("radar",e);k.debug(t),q(t)},"parse")},K=l((e,t,r,a)=>{let i=a.db,s=i.getAxes(),n=i.getCurves(),o=i.getOptions(),p=i.getConfig(),c=i.getDiagramTitle(),m=Q(B(t),p),d=o.max??Math.max(...n.map($=>Math.max(...$.entries))),u=o.min,f=Math.min(p.width,p.height)/2;X(m,s,f,o.ticks,o.graticule),Y(m,s,f,p),M(m,s,n,u,d,o.graticule,p),A(m,n,o.showLegend,p),m.append("text").attr("class","radarTitle").text(c).attr("x",0).attr("y",-p.height/2-p.marginTop)},"draw"),Q=l((e,t)=>{let r=t.width+t.marginLeft+t.marginRight,a=t.height+t.marginTop+t.marginBottom,i={x:t.marginLeft+t.width/2,y:t.marginTop+t.height/2};return e.attr("viewbox",`0 0 ${r} ${a}`).attr("width",r).attr("height",a),e.append("g").attr("transform",`translate(${i.x}, ${i.y})`)},"drawFrame"),X=l((e,t,r,a,i)=>{if(i==="circle")for(let s=0;s<a;s++){let n=r*(s+1)/a;e.append("circle").attr("r",n).attr("class","radarGraticule")}else if(i==="polygon"){let s=t.length;for(let n=0;n<a;n++){let o=r*(n+1)/a,p=t.map((c,m)=>{let d=2*m*Math.PI/s-Math.PI/2;return`${o*Math.cos(d)},${o*Math.sin(d)}`}).join(" ");e.append("polygon").attr("points",p).attr("class","radarGraticule")}}},"drawGraticule"),Y=l((e,t,r,a)=>{let i=t.length;for(let s=0;s<i;s++){let n=t[s].label,o=2*s*Math.PI/i-Math.PI/2;e.append("line").attr("x1",0).attr("y1",0).attr("x2",r*a.axisScaleFactor*Math.cos(o)).attr("y2",r*a.axisScaleFactor*Math.sin(o)).attr("class","radarAxisLine"),e.append("text").text(n).attr("x",r*a.axisLabelFactor*Math.cos(o)).attr("y",r*a.axisLabelFactor*Math.sin(o)).attr("class","radarAxisLabel")}},"drawAxes");function M(e,t,r,a,i,s,n){let o=t.length,p=Math.min(n.width,n.height)/2;r.forEach((c,m)=>{if(c.entries.length!==o)return;let d=c.entries.map((u,f)=>{let $=2*Math.PI*f/o-Math.PI/2,v=L(u,a,i,p);return{x:v*Math.cos($),y:v*Math.sin($)}});s==="circle"?e.append("path").attr("d",T(d,n.curveTension)).attr("class",`radarCurve-${m}`):s==="polygon"&&e.append("polygon").attr("points",d.map(u=>`${u.x},${u.y}`).join(" ")).attr("class",`radarCurve-${m}`)})}l(M,"drawCurves");function L(e,t,r,a){return a*(Math.min(Math.max(e,t),r)-t)/(r-t)}l(L,"relativeRadius");function T(e,t){let r=e.length,a=`M${e[0].x},${e[0].y}`;for(let i=0;i<r;i++){let s=e[(i-1+r)%r],n=e[i],o=e[(i+1)%r],p=e[(i+2)%r],c={x:n.x+(o.x-s.x)*t,y:n.y+(o.y-s.y)*t},m={x:o.x-(p.x-n.x)*t,y:o.y-(p.y-n.y)*t};a+=` C${c.x},${c.y} ${m.x},${m.y} ${o.x},${o.y}`}return`${a} Z`}l(T,"closedRoundCurve");function A(e,t,r,a){if(!r)return;let i=(a.width/2+a.marginRight)*3/4,s=-(a.height/2+a.marginTop)*3/4;t.forEach((n,o)=>{let p=e.append("g").attr("transform",`translate(${i}, ${s+o*20})`);p.append("rect").attr("width",12).attr("height",12).attr("class",`radarLegendBox-${o}`),p.append("text").attr("x",16).attr("y",0).attr("class","radarLegendText").text(n.label)})}l(A,"drawLegend");var tt={draw:K},rt=l((e,t)=>{let r="";for(let a=0;a<e.THEME_COLOR_LIMIT;a++){let i=e[`cScale${a}`];r+=`
		.radarCurve-${a} {
			color: ${i};
			fill: ${i};
			fill-opacity: ${t.curveOpacity};
			stroke: ${i};
			stroke-width: ${t.curveStrokeWidth};
		}
		.radarLegendBox-${a} {
			fill: ${i};
			fill-opacity: ${t.curveOpacity};
			stroke: ${i};
		}
		`}return r},"genIndexStyles"),et=l(e=>{let t=y(I(),w().themeVariables);return{themeVariables:t,radarOptions:y(t.radar,e)}},"buildRadarStyleOptions"),at={parser:J,db:x,renderer:tt,styles:l(({radar:e}={})=>{let{themeVariables:t,radarOptions:r}=et(e);return`
	.radarTitle {
		font-size: ${t.fontSize};
		color: ${t.titleColor};
		dominant-baseline: hanging;
		text-anchor: middle;
	}
	.radarAxisLine {
		stroke: ${r.axisColor};
		stroke-width: ${r.axisStrokeWidth};
	}
	.radarAxisLabel {
		dominant-baseline: middle;
		text-anchor: middle;
		font-size: ${r.axisLabelFontSize}px;
		color: ${r.axisColor};
	}
	.radarGraticule {
		fill: ${r.graticuleColor};
		fill-opacity: ${r.graticuleOpacity};
		stroke: ${r.graticuleColor};
		stroke-width: ${r.graticuleStrokeWidth};
	}
	.radarLegendText {
		text-anchor: start;
		font-size: ${r.legendFontSize}px;
		dominant-baseline: hanging;
	}
	${rt(t,r)}
	`},"styles")};export{at as diagram};
