import{p as B}from"./chunk-ANTBXLJU-ZgvmG8Pb.js";import{_ as s,g as L,s as P,a as V,b as _,q,p as G,l as O,c as I,E as H,I as J,O as K,d as Q,y as U,G as X}from"./mermaid-D59lkToe.js";import{p as Y}from"./treemap-75Q7IDZK-Bh5pVSmo.js";import"./transform-B8bpuzxV.js";import{d as N}from"./arc-ZB5pDULS.js";import{o as Z}from"./ordinal-DDUp3AbE.js";import{a as y,t as z,n as ee}from"./step-BwsUM5iJ.js";import"./index-D4bXoNM3.js";import"./_baseEach-CIMlsWNn.js";import"./_baseUniq-CGK6su7v.js";import"./min-DrLfF3uL.js";import"./_baseMap-1GEe_WcR.js";import"./clone-DFaYgbfI.js";import"./_createAggregator--z161kAx.js";import"./timer-BwIYMJWC.js";import"./init-DLRA0X12.js";function te(e,r){return r<e?-1:r>e?1:r>=e?0:NaN}function ae(e){return e}var ne=X.pie,F={sections:new Map,showData:!1},k=F.sections,R=F.showData,re=structuredClone(ne),j={getConfig:s(()=>structuredClone(re),"getConfig"),clear:s(()=>{k=new Map,R=F.showData,U()},"clear"),setDiagramTitle:G,getDiagramTitle:q,setAccTitle:_,getAccTitle:V,setAccDescription:P,getAccDescription:L,addSection:s(({label:e,value:r})=>{if(r<0)throw new Error(`"${e}" has invalid value: ${r}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);k.has(e)||(k.set(e,r),O.debug(`added new section: ${e}, with value: ${r}`))},"addSection"),getSections:s(()=>k,"getSections"),setShowData:s(e=>{R=e},"setShowData"),getShowData:s(()=>R,"getShowData")},ie=s((e,r)=>{B(e,r),r.setShowData(e.showData),e.sections.map(r.addSection)},"populateDb"),le={parse:s(async e=>{const r=await Y("pie",e);O.debug(r),ie(r,j)},"parse")},oe=s(e=>`
  .pieCircle{
    stroke: ${e.pieStrokeColor};
    stroke-width : ${e.pieStrokeWidth};
    opacity : ${e.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${e.pieOuterStrokeColor};
    stroke-width: ${e.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${e.pieTitleTextSize};
    fill: ${e.pieTitleTextColor};
    font-family: ${e.fontFamily};
  }
  .slice {
    font-family: ${e.fontFamily};
    fill: ${e.pieSectionTextColor};
    font-size:${e.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${e.pieLegendTextColor};
    font-family: ${e.fontFamily};
    font-size: ${e.pieLegendTextSize};
  }
`,"getStyles"),se=s(e=>{const r=[...e.values()].reduce((i,o)=>i+o,0),W=[...e.entries()].map(([i,o])=>({label:i,value:o})).filter(i=>i.value/r*100>=1).sort((i,o)=>o.value-i.value);return function(){var i=ae,o=te,p=null,S=y(0),m=y(z),T=y(0);function l(t){var a,f,D,c,h,u=(t=ee(t)).length,x=0,v=new Array(u),d=new Array(u),g=+S.apply(this,arguments),$=Math.min(z,Math.max(-z,m.apply(this,arguments)-g)),b=Math.min(Math.abs($)/u,T.apply(this,arguments)),C=b*($<0?-1:1);for(a=0;a<u;++a)(h=d[v[a]=a]=+i(t[a],a,t))>0&&(x+=h);for(o!=null?v.sort(function(w,A){return o(d[w],d[A])}):p!=null&&v.sort(function(w,A){return p(t[w],t[A])}),a=0,D=x?($-u*C)/x:0;a<u;++a,g=c)f=v[a],c=g+((h=d[f])>0?h*D:0)+C,d[f]={data:t[f],index:a,value:h,startAngle:g,endAngle:c,padAngle:b};return d}return l.value=function(t){return arguments.length?(i=typeof t=="function"?t:y(+t),l):i},l.sortValues=function(t){return arguments.length?(o=t,p=null,l):o},l.sort=function(t){return arguments.length?(p=t,o=null,l):p},l.startAngle=function(t){return arguments.length?(S=typeof t=="function"?t:y(+t),l):S},l.endAngle=function(t){return arguments.length?(m=typeof t=="function"?t:y(+t),l):m},l.padAngle=function(t){return arguments.length?(T=typeof t=="function"?t:y(+t),l):T},l}().value(i=>i.value)(W)},"createPieArcs"),pe={parser:le,db:j,renderer:{draw:s((e,r,W,E)=>{O.debug(`rendering pie chart
`+e);const i=E.db,o=I(),p=H(i.getConfig(),o.pie),S=18,m=450,T=m,l=J(r),t=l.append("g");t.attr("transform","translate(225,225)");const{themeVariables:a}=o;let[f]=K(a.pieOuterStrokeWidth);f??(f=2);const D=p.textPosition,c=Math.min(T,m)/2-40,h=N().innerRadius(0).outerRadius(c),u=N().innerRadius(c*D).outerRadius(c*D);t.append("circle").attr("cx",0).attr("cy",0).attr("r",c+f/2).attr("class","pieOuterCircle");const x=i.getSections(),v=se(x),d=[a.pie1,a.pie2,a.pie3,a.pie4,a.pie5,a.pie6,a.pie7,a.pie8,a.pie9,a.pie10,a.pie11,a.pie12];let g=0;x.forEach(n=>{g+=n});const $=v.filter(n=>(n.data.value/g*100).toFixed(0)!=="0"),b=Z(d);t.selectAll("mySlices").data($).enter().append("path").attr("d",h).attr("fill",n=>b(n.data.label)).attr("class","pieCircle"),t.selectAll("mySlices").data($).enter().append("text").text(n=>(n.data.value/g*100).toFixed(0)+"%").attr("transform",n=>"translate("+u.centroid(n)+")").style("text-anchor","middle").attr("class","slice"),t.append("text").text(i.getDiagramTitle()).attr("x",0).attr("y",-200).attr("class","pieTitleText");const C=[...x.entries()].map(([n,M])=>({label:n,value:M})),w=t.selectAll(".legend").data(C).enter().append("g").attr("class","legend").attr("transform",(n,M)=>"translate(216,"+(22*M-22*C.length/2)+")");w.append("rect").attr("width",S).attr("height",S).style("fill",n=>b(n.label)).style("stroke",n=>b(n.label)),w.append("text").attr("x",22).attr("y",14).text(n=>i.getShowData()?`${n.label} [${n.value}]`:n.label);const A=512+Math.max(...w.selectAll("text").nodes().map(n=>(n==null?void 0:n.getBoundingClientRect().width)??0));l.attr("viewBox",`0 0 ${A} 450`),Q(l,m,A,p.useMaxWidth)},"draw")},styles:oe};export{pe as diagram};
