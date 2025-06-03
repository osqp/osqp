import{a as ut,g as J,f as pt,d as yt}from"./chunk-D6G4REZN-CaJf0pLm.js";import{_ as s,g as dt,s as ft,a as gt,b as mt,q as xt,p as kt,c as A,d as _t,y as bt}from"./mermaid-Da56-Xo5.js";import{s as K}from"./transform-Cyp0GDF-.js";import{d as Q}from"./arc-Cuwikxov.js";import"./index-aRpq2G87.js";import"./step-BwsUM5iJ.js";import"./timer-DFzT7np-.js";var z=function(){var t=s(function(n,p,a,u){for(a=a||{},u=n.length;u--;a[n[u]]=p);return a},"o"),e=[6,8,10,11,12,14,16,17,18],r=[1,9],l=[1,10],i=[1,11],o=[1,12],h=[1,13],d=[1,14],y={trace:s(function(){},"trace"),yy:{},symbols_:{error:2,start:3,journey:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NEWLINE:10,title:11,acc_title:12,acc_title_value:13,acc_descr:14,acc_descr_value:15,acc_descr_multiline_value:16,section:17,taskName:18,taskData:19,$accept:0,$end:1},terminals_:{2:"error",4:"journey",6:"EOF",8:"SPACE",10:"NEWLINE",11:"title",12:"acc_title",13:"acc_title_value",14:"acc_descr",15:"acc_descr_value",16:"acc_descr_multiline_value",17:"section",18:"taskName",19:"taskData"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,2]],performAction:s(function(n,p,a,u,f,c,v){var x=c.length-1;switch(f){case 1:return c[x-1];case 2:case 6:case 7:this.$=[];break;case 3:c[x-1].push(c[x]),this.$=c[x-1];break;case 4:case 5:this.$=c[x];break;case 8:u.setDiagramTitle(c[x].substr(6)),this.$=c[x].substr(6);break;case 9:this.$=c[x].trim(),u.setAccTitle(this.$);break;case 10:case 11:this.$=c[x].trim(),u.setAccDescription(this.$);break;case 12:u.addSection(c[x].substr(8)),this.$=c[x].substr(8);break;case 13:u.addTask(c[x-1],c[x]),this.$="task"}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:r,12:l,14:i,16:o,17:h,18:d},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:15,11:r,12:l,14:i,16:o,17:h,18:d},t(e,[2,5]),t(e,[2,6]),t(e,[2,8]),{13:[1,16]},{15:[1,17]},t(e,[2,11]),t(e,[2,12]),{19:[1,18]},t(e,[2,4]),t(e,[2,9]),t(e,[2,10]),t(e,[2,13])],defaultActions:{},parseError:s(function(n,p){if(!p.recoverable){var a=new Error(n);throw a.hash=p,a}this.trace(n)},"parseError"),parse:s(function(n){var p=this,a=[0],u=[],f=[null],c=[],v=this.table,x="",L=0,X=0,lt=c.slice.call(arguments,1),k=Object.create(this.lexer),M={yy:{}};for(var N in this.yy)Object.prototype.hasOwnProperty.call(this.yy,N)&&(M.yy[N]=this.yy[N]);k.setInput(n,M.yy),M.yy.lexer=k,M.yy.parser=this,k.yylloc===void 0&&(k.yylloc={});var R=k.yylloc;c.push(R);var ht=k.options&&k.options.ranges;function G(){var w;return typeof(w=u.pop()||k.lex()||1)!="number"&&(w instanceof Array&&(w=(u=w).pop()),w=p.symbols_[w]||w),w}typeof M.yy.parseError=="function"?this.parseError=M.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError,s(function(w){a.length=a.length-2*w,f.length=f.length-w,c.length=c.length-w},"popStack"),s(G,"lex");for(var _,E,b,U,O,T,H,B,P={};;){if(E=a[a.length-1],this.defaultActions[E]?b=this.defaultActions[E]:(_==null&&(_=G()),b=v[E]&&v[E][_]),b===void 0||!b.length||!b[0]){var Z="";for(O in B=[],v[E])this.terminals_[O]&&O>2&&B.push("'"+this.terminals_[O]+"'");Z=k.showPosition?"Parse error on line "+(L+1)+`:
`+k.showPosition()+`
Expecting `+B.join(", ")+", got '"+(this.terminals_[_]||_)+"'":"Parse error on line "+(L+1)+": Unexpected "+(_==1?"end of input":"'"+(this.terminals_[_]||_)+"'"),this.parseError(Z,{text:k.match,token:this.terminals_[_]||_,line:k.yylineno,loc:R,expected:B})}if(b[0]instanceof Array&&b.length>1)throw new Error("Parse Error: multiple actions possible at state: "+E+", token: "+_);switch(b[0]){case 1:a.push(_),f.push(k.yytext),c.push(k.yylloc),a.push(b[1]),_=null,X=k.yyleng,x=k.yytext,L=k.yylineno,R=k.yylloc;break;case 2:if(T=this.productions_[b[1]][1],P.$=f[f.length-T],P._$={first_line:c[c.length-(T||1)].first_line,last_line:c[c.length-1].last_line,first_column:c[c.length-(T||1)].first_column,last_column:c[c.length-1].last_column},ht&&(P._$.range=[c[c.length-(T||1)].range[0],c[c.length-1].range[1]]),(U=this.performAction.apply(P,[x,X,L,M.yy,b[1],f,c].concat(lt)))!==void 0)return U;T&&(a=a.slice(0,-1*T*2),f=f.slice(0,-1*T),c=c.slice(0,-1*T)),a.push(this.productions_[b[1]][0]),f.push(P.$),c.push(P._$),H=v[a[a.length-2]][a[a.length-1]],a.push(H);break;case 3:return!0}}return!0},"parse")},m=function(){return{EOF:1,parseError:s(function(n,p){if(!this.yy.parser)throw new Error(n);this.yy.parser.parseError(n,p)},"parseError"),setInput:s(function(n,p){return this.yy=p||this.yy||{},this._input=n,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:s(function(){var n=this._input[0];return this.yytext+=n,this.yyleng++,this.offset++,this.match+=n,this.matched+=n,n.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),n},"input"),unput:s(function(n){var p=n.length,a=n.split(/(?:\r\n?|\n)/g);this._input=n+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-p),this.offset-=p;var u=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),a.length-1&&(this.yylineno-=a.length-1);var f=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:a?(a.length===u.length?this.yylloc.first_column:0)+u[u.length-a.length].length-a[0].length:this.yylloc.first_column-p},this.options.ranges&&(this.yylloc.range=[f[0],f[0]+this.yyleng-p]),this.yyleng=this.yytext.length,this},"unput"),more:s(function(){return this._more=!0,this},"more"),reject:s(function(){return this.options.backtrack_lexer?(this._backtrack=!0,this):this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"reject"),less:s(function(n){this.unput(this.match.slice(n))},"less"),pastInput:s(function(){var n=this.matched.substr(0,this.matched.length-this.match.length);return(n.length>20?"...":"")+n.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:s(function(){var n=this.match;return n.length<20&&(n+=this._input.substr(0,20-n.length)),(n.substr(0,20)+(n.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:s(function(){var n=this.pastInput(),p=new Array(n.length+1).join("-");return n+this.upcomingInput()+`
`+p+"^"},"showPosition"),test_match:s(function(n,p){var a,u,f;if(this.options.backtrack_lexer&&(f={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(f.yylloc.range=this.yylloc.range.slice(0))),(u=n[0].match(/(?:\r\n?|\n).*/g))&&(this.yylineno+=u.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:u?u[u.length-1].length-u[u.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+n[0].length},this.yytext+=n[0],this.match+=n[0],this.matches=n,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(n[0].length),this.matched+=n[0],a=this.performAction.call(this,this.yy,this,p,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),a)return a;if(this._backtrack){for(var c in f)this[c]=f[c];return!1}return!1},"test_match"),next:s(function(){if(this.done)return this.EOF;var n,p,a,u;this._input||(this.done=!0),this._more||(this.yytext="",this.match="");for(var f=this._currentRules(),c=0;c<f.length;c++)if((a=this._input.match(this.rules[f[c]]))&&(!p||a[0].length>p[0].length)){if(p=a,u=c,this.options.backtrack_lexer){if((n=this.test_match(a,f[c]))!==!1)return n;if(this._backtrack){p=!1;continue}return!1}if(!this.options.flex)break}return p?(n=this.test_match(p,f[u]))!==!1&&n:this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:s(function(){var n=this.next();return n||this.lex()},"lex"),begin:s(function(n){this.conditionStack.push(n)},"begin"),popState:s(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:s(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:s(function(n){return(n=this.conditionStack.length-1-Math.abs(n||0))>=0?this.conditionStack[n]:"INITIAL"},"topState"),pushState:s(function(n){this.begin(n)},"pushState"),stateStackSize:s(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:s(function(n,p,a,u){switch(a){case 0:case 1:case 3:case 4:break;case 2:return 10;case 5:return 4;case 6:return 11;case 7:return this.begin("acc_title"),12;case 8:return this.popState(),"acc_title_value";case 9:return this.begin("acc_descr"),14;case 10:return this.popState(),"acc_descr_value";case 11:this.begin("acc_descr_multiline");break;case 12:this.popState();break;case 13:return"acc_descr_multiline_value";case 14:return 17;case 15:return 18;case 16:return 19;case 17:return":";case 18:return 6;case 19:return"INVALID"}},"anonymous"),rules:[/^(?:%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:#[^\n]*)/i,/^(?:journey\b)/i,/^(?:title\s[^#\n;]+)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:section\s[^#:\n;]+)/i,/^(?:[^#:\n;]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[12,13],inclusive:!1},acc_descr:{rules:[10],inclusive:!1},acc_title:{rules:[8],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,9,11,14,15,16,17,18,19],inclusive:!0}}}}();function g(){this.yy={}}return y.lexer=m,s(g,"Parser"),g.prototype=y,y.Parser=g,new g}();z.parser=z;var wt=z,C="",Y=[],j=[],V=[],$t=s(function(){Y.length=0,j.length=0,C="",V.length=0,bt()},"clear"),vt=s(function(t){C=t,Y.push(t)},"addSection"),Tt=s(function(){return Y},"getSections"),St=s(function(){let t=tt(),e=0;for(;!t&&e<100;)t=tt(),e++;return j.push(...V),j},"getTasks"),Mt=s(function(){const t=[];return j.forEach(e=>{e.people&&t.push(...e.people)}),[...new Set(t)].sort()},"updateActors"),Et=s(function(t,e){const r=e.substr(1).split(":");let l=0,i=[];r.length===1?(l=Number(r[0]),i=[]):(l=Number(r[0]),i=r[1].split(","));const o=i.map(d=>d.trim()),h={section:C,type:C,people:o,task:t,score:l};V.push(h)},"addTask"),At=s(function(t){const e={section:C,type:C,description:t,task:t,classes:[]};j.push(e)},"addTaskOrg"),tt=s(function(){const t=s(function(r){return V[r].processed},"compileTask");let e=!0;for(const[r,l]of V.entries())t(r),e=e&&l.processed;return e},"compileTasks"),It=s(function(){return Mt()},"getActors"),et={getConfig:s(()=>A().journey,"getConfig"),clear:$t,setDiagramTitle:kt,getDiagramTitle:xt,setAccTitle:mt,getAccTitle:gt,setAccDescription:ft,getAccDescription:dt,addSection:vt,getSections:Tt,getTasks:St,addTask:Et,addTaskOrg:At,getActors:It},Pt=s(t=>`.label {
    font-family: ${t.fontFamily};
    color: ${t.textColor};
  }
  .mouth {
    stroke: #666;
  }

  line {
    stroke: ${t.textColor}
  }

  .legend {
    fill: ${t.textColor};
    font-family: ${t.fontFamily};
  }

  .label text {
    fill: #333;
  }
  .label {
    color: ${t.textColor}
  }

  .face {
    ${t.faceColor?`fill: ${t.faceColor}`:"fill: #FFF8DC"};
    stroke: #999;
  }

  .node rect,
  .node circle,
  .node ellipse,
  .node polygon,
  .node path {
    fill: ${t.mainBkg};
    stroke: ${t.nodeBorder};
    stroke-width: 1px;
  }

  .node .label {
    text-align: center;
  }
  .node.clickable {
    cursor: pointer;
  }

  .arrowheadPath {
    fill: ${t.arrowheadColor};
  }

  .edgePath .path {
    stroke: ${t.lineColor};
    stroke-width: 1.5px;
  }

  .flowchart-link {
    stroke: ${t.lineColor};
    fill: none;
  }

  .edgeLabel {
    background-color: ${t.edgeLabelBackground};
    rect {
      opacity: 0.5;
    }
    text-align: center;
  }

  .cluster rect {
  }

  .cluster text {
    fill: ${t.titleColor};
  }

  div.mermaidTooltip {
    position: absolute;
    text-align: center;
    max-width: 200px;
    padding: 2px;
    font-family: ${t.fontFamily};
    font-size: 12px;
    background: ${t.tertiaryColor};
    border: 1px solid ${t.border2};
    border-radius: 2px;
    pointer-events: none;
    z-index: 100;
  }

  .task-type-0, .section-type-0  {
    ${t.fillType0?`fill: ${t.fillType0}`:""};
  }
  .task-type-1, .section-type-1  {
    ${t.fillType0?`fill: ${t.fillType1}`:""};
  }
  .task-type-2, .section-type-2  {
    ${t.fillType0?`fill: ${t.fillType2}`:""};
  }
  .task-type-3, .section-type-3  {
    ${t.fillType0?`fill: ${t.fillType3}`:""};
  }
  .task-type-4, .section-type-4  {
    ${t.fillType0?`fill: ${t.fillType4}`:""};
  }
  .task-type-5, .section-type-5  {
    ${t.fillType0?`fill: ${t.fillType5}`:""};
  }
  .task-type-6, .section-type-6  {
    ${t.fillType0?`fill: ${t.fillType6}`:""};
  }
  .task-type-7, .section-type-7  {
    ${t.fillType0?`fill: ${t.fillType7}`:""};
  }

  .actor-0 {
    ${t.actor0?`fill: ${t.actor0}`:""};
  }
  .actor-1 {
    ${t.actor1?`fill: ${t.actor1}`:""};
  }
  .actor-2 {
    ${t.actor2?`fill: ${t.actor2}`:""};
  }
  .actor-3 {
    ${t.actor3?`fill: ${t.actor3}`:""};
  }
  .actor-4 {
    ${t.actor4?`fill: ${t.actor4}`:""};
  }
  .actor-5 {
    ${t.actor5?`fill: ${t.actor5}`:""};
  }
`,"getStyles"),q=s(function(t,e){return yt(t,e)},"drawRect"),Ct=s(function(t,e){const l=t.append("circle").attr("cx",e.cx).attr("cy",e.cy).attr("class","face").attr("r",15).attr("stroke-width",2).attr("overflow","visible"),i=t.append("g");function o(y){const m=Q().startAngle(Math.PI/2).endAngle(Math.PI/2*3).innerRadius(7.5).outerRadius(6.8181818181818175);y.append("path").attr("class","mouth").attr("d",m).attr("transform","translate("+e.cx+","+(e.cy+2)+")")}function h(y){const m=Q().startAngle(3*Math.PI/2).endAngle(Math.PI/2*5).innerRadius(7.5).outerRadius(6.8181818181818175);y.append("path").attr("class","mouth").attr("d",m).attr("transform","translate("+e.cx+","+(e.cy+7)+")")}function d(y){y.append("line").attr("class","mouth").attr("stroke",2).attr("x1",e.cx-5).attr("y1",e.cy+7).attr("x2",e.cx+5).attr("y2",e.cy+7).attr("class","mouth").attr("stroke-width","1px").attr("stroke","#666")}return i.append("circle").attr("cx",e.cx-5).attr("cy",e.cy-5).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666"),i.append("circle").attr("cx",e.cx+5).attr("cy",e.cy-5).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666"),s(o,"smile"),s(h,"sad"),s(d,"ambivalent"),e.score>3?o(i):e.score<3?h(i):d(i),l},"drawFace"),nt=s(function(t,e){const r=t.append("circle");return r.attr("cx",e.cx),r.attr("cy",e.cy),r.attr("class","actor-"+e.pos),r.attr("fill",e.fill),r.attr("stroke",e.stroke),r.attr("r",e.r),r.class!==void 0&&r.attr("class",r.class),e.title!==void 0&&r.append("title").text(e.title),r},"drawCircle"),it=s(function(t,e){return pt(t,e)},"drawText"),jt=s(function(t,e){function r(i,o,h,d,y){return i+","+o+" "+(i+h)+","+o+" "+(i+h)+","+(o+d-y)+" "+(i+h-1.2*y)+","+(o+d)+" "+i+","+(o+d)}s(r,"genPoints");const l=t.append("polygon");l.attr("points",r(e.x,e.y,50,20,7)),l.attr("class","labelBox"),e.y=e.y+e.labelMargin,e.x=e.x+.5*e.labelMargin,it(t,e)},"drawLabel"),Vt=s(function(t,e,r){const l=t.append("g"),i=J();i.x=e.x,i.y=e.y,i.fill=e.fill,i.width=r.width*e.taskCount+r.diagramMarginX*(e.taskCount-1),i.height=r.height,i.class="journey-section section-type-"+e.num,i.rx=3,i.ry=3,q(l,i),at(r)(e.text,l,i.x,i.y,i.width,i.height,{class:"journey-section section-type-"+e.num},r,e.colour)},"drawSection"),st=-1,Ft=s(function(t,e,r){const l=e.x+r.width/2,i=t.append("g");st++,i.append("line").attr("id","task"+st).attr("x1",l).attr("y1",e.y).attr("x2",l).attr("y2",450).attr("class","task-line").attr("stroke-width","1px").attr("stroke-dasharray","4 2").attr("stroke","#666"),Ct(i,{cx:l,cy:300+30*(5-e.score),score:e.score});const o=J();o.x=e.x,o.y=e.y,o.fill=e.fill,o.width=r.width,o.height=r.height,o.class="task task-type-"+e.num,o.rx=3,o.ry=3,q(i,o);let h=e.x+14;e.people.forEach(d=>{const y=e.actors[d].color,m={cx:h,cy:e.y,r:7,fill:y,stroke:"#000",title:d,pos:e.actors[d].position};nt(i,m),h+=10}),at(r)(e.task,i,o.x,o.y,o.width,o.height,{class:"task"},r,e.colour)},"drawTask"),Lt=s(function(t,e){ut(t,e)},"drawBackgroundRect"),at=function(){function t(i,o,h,d,y,m,g,n){l(o.append("text").attr("x",h+y/2).attr("y",d+m/2+5).style("font-color",n).style("text-anchor","middle").text(i),g)}function e(i,o,h,d,y,m,g,n,p){const{taskFontSize:a,taskFontFamily:u}=n,f=i.split(/<br\s*\/?>/gi);for(let c=0;c<f.length;c++){const v=c*a-a*(f.length-1)/2,x=o.append("text").attr("x",h+y/2).attr("y",d).attr("fill",p).style("text-anchor","middle").style("font-size",a).style("font-family",u);x.append("tspan").attr("x",h+y/2).attr("dy",v).text(f[c]),x.attr("y",d+m/2).attr("dominant-baseline","central").attr("alignment-baseline","central"),l(x,g)}}function r(i,o,h,d,y,m,g,n){const p=o.append("switch"),a=p.append("foreignObject").attr("x",h).attr("y",d).attr("width",y).attr("height",m).attr("position","fixed").append("xhtml:div").style("display","table").style("height","100%").style("width","100%");a.append("div").attr("class","label").style("display","table-cell").style("text-align","center").style("vertical-align","middle").text(i),e(i,p,h,d,y,m,g,n),l(a,g)}function l(i,o){for(const h in o)h in o&&i.attr(h,o[h])}return s(t,"byText"),s(e,"byTspan"),s(r,"byFo"),s(l,"_setTextAttrs"),function(i){return i.textPlacement==="fo"?r:i.textPlacement==="old"?t:e}}(),F={drawRect:q,drawCircle:nt,drawSection:Vt,drawText:it,drawLabel:jt,drawTask:Ft,drawBackgroundRect:Lt,initGraphics:s(function(t){t.append("defs").append("marker").attr("id","arrowhead").attr("refX",5).attr("refY",2).attr("markerWidth",6).attr("markerHeight",4).attr("orient","auto").append("path").attr("d","M 0,0 V 4 L6,2 Z")},"initGraphics")},Ot=s(function(t){Object.keys(t).forEach(function(e){D[e]=t[e]})},"setConf"),S={};function rt(t){const e=A().journey;let r=60;Object.keys(S).forEach(l=>{const i=S[l].color,o={cx:20,cy:r,r:7,fill:i,stroke:"#000",pos:S[l].position};F.drawCircle(t,o);const h={x:40,y:r+7,fill:"#666",text:l,textMargin:5|e.boxTextMargin};F.drawText(t,h),r+=20})}s(rt,"drawActorLegend");var D=A().journey,I=D.leftMargin,Bt=s(function(t,e,r,l){const i=A().journey,o=A().securityLevel;let h;o==="sandbox"&&(h=K("#i"+e));const d=K(o==="sandbox"?h.nodes()[0].contentDocument.body:"body");$.init();const y=d.select("#"+e);F.initGraphics(y);const m=l.db.getTasks(),g=l.db.getDiagramTitle(),n=l.db.getActors();for(const v in S)delete S[v];let p=0;n.forEach(v=>{S[v]={color:i.actorColours[p%i.actorColours.length],position:p},p++}),rt(y),$.insert(0,0,I,50*Object.keys(S).length),Dt(y,m,0);const a=$.getBounds();g&&y.append("text").text(g).attr("x",I).attr("font-size","4ex").attr("font-weight","bold").attr("y",25);const u=a.stopy-a.starty+2*i.diagramMarginY,f=I+a.stopx+2*i.diagramMarginX;_t(y,u,f,i.useMaxWidth),y.append("line").attr("x1",I).attr("y1",4*i.height).attr("x2",f-I-4).attr("y2",4*i.height).attr("stroke-width",4).attr("stroke","black").attr("marker-end","url(#arrowhead)");const c=g?70:0;y.attr("viewBox",`${a.startx} -25 ${f} ${u+c}`),y.attr("preserveAspectRatio","xMinYMin meet"),y.attr("height",u+c+25)},"draw"),$={data:{startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},verticalPos:0,sequenceItems:[],init:s(function(){this.sequenceItems=[],this.data={startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},this.verticalPos=0},"init"),updateVal:s(function(t,e,r,l){t[e]===void 0?t[e]=r:t[e]=l(r,t[e])},"updateVal"),updateBounds:s(function(t,e,r,l){const i=A().journey,o=this;let h=0;function d(y){return s(function(m){h++;const g=o.sequenceItems.length-h+1;o.updateVal(m,"starty",e-g*i.boxMargin,Math.min),o.updateVal(m,"stopy",l+g*i.boxMargin,Math.max),o.updateVal($.data,"startx",t-g*i.boxMargin,Math.min),o.updateVal($.data,"stopx",r+g*i.boxMargin,Math.max),y!=="activation"&&(o.updateVal(m,"startx",t-g*i.boxMargin,Math.min),o.updateVal(m,"stopx",r+g*i.boxMargin,Math.max),o.updateVal($.data,"starty",e-g*i.boxMargin,Math.min),o.updateVal($.data,"stopy",l+g*i.boxMargin,Math.max))},"updateItemBounds")}s(d,"updateFn"),this.sequenceItems.forEach(d())},"updateBounds"),insert:s(function(t,e,r,l){const i=Math.min(t,r),o=Math.max(t,r),h=Math.min(e,l),d=Math.max(e,l);this.updateVal($.data,"startx",i,Math.min),this.updateVal($.data,"starty",h,Math.min),this.updateVal($.data,"stopx",o,Math.max),this.updateVal($.data,"stopy",d,Math.max),this.updateBounds(i,h,o,d)},"insert"),bumpVerticalPos:s(function(t){this.verticalPos=this.verticalPos+t,this.data.stopy=this.verticalPos},"bumpVerticalPos"),getVerticalPos:s(function(){return this.verticalPos},"getVerticalPos"),getBounds:s(function(){return this.data},"getBounds")},W=D.sectionFills,ot=D.sectionColours,Dt=s(function(t,e,r){const l=A().journey;let i="";const o=r+(2*l.height+l.diagramMarginY);let h=0,d="#CCC",y="black",m=0;for(const[g,n]of e.entries()){if(i!==n.section){d=W[h%W.length],m=h%W.length,y=ot[h%ot.length];let a=0;const u=n.section;for(let c=g;c<e.length&&e[c].section==u;c++)a+=1;const f={x:g*l.taskMargin+g*l.width+I,y:50,text:n.section,fill:d,num:m,colour:y,taskCount:a};F.drawSection(t,f,l),i=n.section,h++}const p=n.people.reduce((a,u)=>(S[u]&&(a[u]=S[u]),a),{});n.x=g*l.taskMargin+g*l.width+I,n.y=o,n.width=l.diagramMarginX,n.height=l.diagramMarginY,n.colour=y,n.fill=d,n.num=m,n.actors=p,F.drawTask(t,n,l),$.insert(n.x,n.y,n.x+n.width+l.taskMargin,450)}},"drawTasks"),ct={setConf:Ot,draw:Bt},Nt={parser:wt,db:et,renderer:ct,styles:Pt,init:s(t=>{ct.setConf(t.journey),et.clear()},"init")};export{Nt as diagram};
