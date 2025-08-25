import{a as pt,g as K,f as yt,d as dt}from"./chunk-OMD6QJNC-CwDxph2q.js";import{g as ft}from"./chunk-GLLZNHP4-DRBUPzV6.js";import{_ as s,g as gt,s as mt,a as xt,b as kt,q as bt,p as _t,c as F,d as wt,y as vt}from"./mermaid-D59lkToe.js";import{s as Q}from"./transform-B8bpuzxV.js";import{d as tt}from"./arc-ZB5pDULS.js";import"./index-D4bXoNM3.js";import"./step-BwsUM5iJ.js";import"./timer-BwIYMJWC.js";var q=function(){var t=s(function(n,u,a,p){for(a=a||{},p=n.length;p--;a[n[p]]=u);return a},"o"),e=[6,8,10,11,12,14,16,17,18],r=[1,9],c=[1,10],i=[1,11],l=[1,12],y=[1,13],h=[1,14],f={trace:s(function(){},"trace"),yy:{},symbols_:{error:2,start:3,journey:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NEWLINE:10,title:11,acc_title:12,acc_title_value:13,acc_descr:14,acc_descr_value:15,acc_descr_multiline_value:16,section:17,taskName:18,taskData:19,$accept:0,$end:1},terminals_:{2:"error",4:"journey",6:"EOF",8:"SPACE",10:"NEWLINE",11:"title",12:"acc_title",13:"acc_title_value",14:"acc_descr",15:"acc_descr_value",16:"acc_descr_multiline_value",17:"section",18:"taskName",19:"taskData"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,2]],performAction:s(function(n,u,a,p,d,o,$){var x=o.length-1;switch(d){case 1:return o[x-1];case 2:case 6:case 7:this.$=[];break;case 3:o[x-1].push(o[x]),this.$=o[x-1];break;case 4:case 5:this.$=o[x];break;case 8:p.setDiagramTitle(o[x].substr(6)),this.$=o[x].substr(6);break;case 9:this.$=o[x].trim(),p.setAccTitle(this.$);break;case 10:case 11:this.$=o[x].trim(),p.setAccDescription(this.$);break;case 12:p.addSection(o[x].substr(8)),this.$=o[x].substr(8);break;case 13:p.addTask(o[x-1],o[x]),this.$="task"}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:r,12:c,14:i,16:l,17:y,18:h},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:15,11:r,12:c,14:i,16:l,17:y,18:h},t(e,[2,5]),t(e,[2,6]),t(e,[2,8]),{13:[1,16]},{15:[1,17]},t(e,[2,11]),t(e,[2,12]),{19:[1,18]},t(e,[2,4]),t(e,[2,9]),t(e,[2,10]),t(e,[2,13])],defaultActions:{},parseError:s(function(n,u){if(!u.recoverable){var a=new Error(n);throw a.hash=u,a}this.trace(n)},"parseError"),parse:s(function(n){var u=this,a=[0],p=[],d=[null],o=[],$=this.table,x="",E=0,P=0,ht=o.slice.call(arguments,1),k=Object.create(this.lexer),A={yy:{}};for(var z in this.yy)Object.prototype.hasOwnProperty.call(this.yy,z)&&(A.yy[z]=this.yy[z]);k.setInput(n,A.yy),A.yy.lexer=k,A.yy.parser=this,k.yylloc===void 0&&(k.yylloc={});var Y=k.yylloc;o.push(Y);var ut=k.options&&k.options.ranges;function U(){var w;return typeof(w=p.pop()||k.lex()||1)!="number"&&(w instanceof Array&&(w=(p=w).pop()),w=u.symbols_[w]||w),w}typeof A.yy.parseError=="function"?this.parseError=A.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError,s(function(w){a.length=a.length-2*w,d.length=d.length-w,o.length=o.length-w},"popStack"),s(U,"lex");for(var b,I,_,H,O,T,Z,D,j={};;){if(I=a[a.length-1],this.defaultActions[I]?_=this.defaultActions[I]:(b==null&&(b=U()),_=$[I]&&$[I][b]),_===void 0||!_.length||!_[0]){var J="";for(O in D=[],$[I])this.terminals_[O]&&O>2&&D.push("'"+this.terminals_[O]+"'");J=k.showPosition?"Parse error on line "+(E+1)+`:
`+k.showPosition()+`
Expecting `+D.join(", ")+", got '"+(this.terminals_[b]||b)+"'":"Parse error on line "+(E+1)+": Unexpected "+(b==1?"end of input":"'"+(this.terminals_[b]||b)+"'"),this.parseError(J,{text:k.match,token:this.terminals_[b]||b,line:k.yylineno,loc:Y,expected:D})}if(_[0]instanceof Array&&_.length>1)throw new Error("Parse Error: multiple actions possible at state: "+I+", token: "+b);switch(_[0]){case 1:a.push(b),d.push(k.yytext),o.push(k.yylloc),a.push(_[1]),b=null,P=k.yyleng,x=k.yytext,E=k.yylineno,Y=k.yylloc;break;case 2:if(T=this.productions_[_[1]][1],j.$=d[d.length-T],j._$={first_line:o[o.length-(T||1)].first_line,last_line:o[o.length-1].last_line,first_column:o[o.length-(T||1)].first_column,last_column:o[o.length-1].last_column},ut&&(j._$.range=[o[o.length-(T||1)].range[0],o[o.length-1].range[1]]),(H=this.performAction.apply(j,[x,P,E,A.yy,_[1],d,o].concat(ht)))!==void 0)return H;T&&(a=a.slice(0,-1*T*2),d=d.slice(0,-1*T),o=o.slice(0,-1*T)),a.push(this.productions_[_[1]][0]),d.push(j.$),o.push(j._$),Z=$[a[a.length-2]][a[a.length-1]],a.push(Z);break;case 3:return!0}}return!0},"parse")},g=function(){return{EOF:1,parseError:s(function(n,u){if(!this.yy.parser)throw new Error(n);this.yy.parser.parseError(n,u)},"parseError"),setInput:s(function(n,u){return this.yy=u||this.yy||{},this._input=n,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:s(function(){var n=this._input[0];return this.yytext+=n,this.yyleng++,this.offset++,this.match+=n,this.matched+=n,n.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),n},"input"),unput:s(function(n){var u=n.length,a=n.split(/(?:\r\n?|\n)/g);this._input=n+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-u),this.offset-=u;var p=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),a.length-1&&(this.yylineno-=a.length-1);var d=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:a?(a.length===p.length?this.yylloc.first_column:0)+p[p.length-a.length].length-a[0].length:this.yylloc.first_column-u},this.options.ranges&&(this.yylloc.range=[d[0],d[0]+this.yyleng-u]),this.yyleng=this.yytext.length,this},"unput"),more:s(function(){return this._more=!0,this},"more"),reject:s(function(){return this.options.backtrack_lexer?(this._backtrack=!0,this):this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"reject"),less:s(function(n){this.unput(this.match.slice(n))},"less"),pastInput:s(function(){var n=this.matched.substr(0,this.matched.length-this.match.length);return(n.length>20?"...":"")+n.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:s(function(){var n=this.match;return n.length<20&&(n+=this._input.substr(0,20-n.length)),(n.substr(0,20)+(n.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:s(function(){var n=this.pastInput(),u=new Array(n.length+1).join("-");return n+this.upcomingInput()+`
`+u+"^"},"showPosition"),test_match:s(function(n,u){var a,p,d;if(this.options.backtrack_lexer&&(d={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(d.yylloc.range=this.yylloc.range.slice(0))),(p=n[0].match(/(?:\r\n?|\n).*/g))&&(this.yylineno+=p.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:p?p[p.length-1].length-p[p.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+n[0].length},this.yytext+=n[0],this.match+=n[0],this.matches=n,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(n[0].length),this.matched+=n[0],a=this.performAction.call(this,this.yy,this,u,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),a)return a;if(this._backtrack){for(var o in d)this[o]=d[o];return!1}return!1},"test_match"),next:s(function(){if(this.done)return this.EOF;var n,u,a,p;this._input||(this.done=!0),this._more||(this.yytext="",this.match="");for(var d=this._currentRules(),o=0;o<d.length;o++)if((a=this._input.match(this.rules[d[o]]))&&(!u||a[0].length>u[0].length)){if(u=a,p=o,this.options.backtrack_lexer){if((n=this.test_match(a,d[o]))!==!1)return n;if(this._backtrack){u=!1;continue}return!1}if(!this.options.flex)break}return u?(n=this.test_match(u,d[p]))!==!1&&n:this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:s(function(){var n=this.next();return n||this.lex()},"lex"),begin:s(function(n){this.conditionStack.push(n)},"begin"),popState:s(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:s(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:s(function(n){return(n=this.conditionStack.length-1-Math.abs(n||0))>=0?this.conditionStack[n]:"INITIAL"},"topState"),pushState:s(function(n){this.begin(n)},"pushState"),stateStackSize:s(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:s(function(n,u,a,p){switch(a){case 0:case 1:case 3:case 4:break;case 2:return 10;case 5:return 4;case 6:return 11;case 7:return this.begin("acc_title"),12;case 8:return this.popState(),"acc_title_value";case 9:return this.begin("acc_descr"),14;case 10:return this.popState(),"acc_descr_value";case 11:this.begin("acc_descr_multiline");break;case 12:this.popState();break;case 13:return"acc_descr_multiline_value";case 14:return 17;case 15:return 18;case 16:return 19;case 17:return":";case 18:return 6;case 19:return"INVALID"}},"anonymous"),rules:[/^(?:%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:#[^\n]*)/i,/^(?:journey\b)/i,/^(?:title\s[^#\n;]+)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:section\s[^#:\n;]+)/i,/^(?:[^#:\n;]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[12,13],inclusive:!1},acc_descr:{rules:[10],inclusive:!1},acc_title:{rules:[8],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,9,11,14,15,16,17,18,19],inclusive:!0}}}}();function m(){this.yy={}}return f.lexer=g,s(m,"Parser"),m.prototype=f,f.Parser=m,new m}();q.parser=q;var $t=q,V="",W=[],B=[],L=[],Mt=s(function(){W.length=0,B.length=0,V="",L.length=0,vt()},"clear"),Tt=s(function(t){V=t,W.push(t)},"addSection"),St=s(function(){return W},"getSections"),Ct=s(function(){let t=et(),e=0;for(;!t&&e<100;)t=et(),e++;return B.push(...L),B},"getTasks"),Et=s(function(){const t=[];return B.forEach(e=>{e.people&&t.push(...e.people)}),[...new Set(t)].sort()},"updateActors"),At=s(function(t,e){const r=e.substr(1).split(":");let c=0,i=[];r.length===1?(c=Number(r[0]),i=[]):(c=Number(r[0]),i=r[1].split(","));const l=i.map(h=>h.trim()),y={section:V,type:V,people:l,task:t,score:c};L.push(y)},"addTask"),It=s(function(t){const e={section:V,type:V,description:t,task:t,classes:[]};B.push(e)},"addTaskOrg"),et=s(function(){const t=s(function(r){return L[r].processed},"compileTask");let e=!0;for(const[r,c]of L.entries())t(r),e=e&&c.processed;return e},"compileTasks"),Pt=s(function(){return Et()},"getActors"),nt={getConfig:s(()=>F().journey,"getConfig"),clear:Mt,setDiagramTitle:_t,getDiagramTitle:bt,setAccTitle:kt,getAccTitle:xt,setAccDescription:mt,getAccDescription:gt,addSection:Tt,getSections:St,getTasks:Ct,addTask:At,addTaskOrg:It,getActors:Pt},jt=s(t=>`.label {
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
  ${ft()}
`,"getStyles"),X=s(function(t,e){return dt(t,e)},"drawRect"),Ft=s(function(t,e){const c=t.append("circle").attr("cx",e.cx).attr("cy",e.cy).attr("class","face").attr("r",15).attr("stroke-width",2).attr("overflow","visible"),i=t.append("g");function l(f){const g=tt().startAngle(Math.PI/2).endAngle(Math.PI/2*3).innerRadius(7.5).outerRadius(6.8181818181818175);f.append("path").attr("class","mouth").attr("d",g).attr("transform","translate("+e.cx+","+(e.cy+2)+")")}function y(f){const g=tt().startAngle(3*Math.PI/2).endAngle(Math.PI/2*5).innerRadius(7.5).outerRadius(6.8181818181818175);f.append("path").attr("class","mouth").attr("d",g).attr("transform","translate("+e.cx+","+(e.cy+7)+")")}function h(f){f.append("line").attr("class","mouth").attr("stroke",2).attr("x1",e.cx-5).attr("y1",e.cy+7).attr("x2",e.cx+5).attr("y2",e.cy+7).attr("class","mouth").attr("stroke-width","1px").attr("stroke","#666")}return i.append("circle").attr("cx",e.cx-5).attr("cy",e.cy-5).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666"),i.append("circle").attr("cx",e.cx+5).attr("cy",e.cy-5).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666"),s(l,"smile"),s(y,"sad"),s(h,"ambivalent"),e.score>3?l(i):e.score<3?y(i):h(i),c},"drawFace"),it=s(function(t,e){const r=t.append("circle");return r.attr("cx",e.cx),r.attr("cy",e.cy),r.attr("class","actor-"+e.pos),r.attr("fill",e.fill),r.attr("stroke",e.stroke),r.attr("r",e.r),r.class!==void 0&&r.attr("class",r.class),e.title!==void 0&&r.append("title").text(e.title),r},"drawCircle"),st=s(function(t,e){return yt(t,e)},"drawText"),Vt=s(function(t,e){function r(i,l,y,h,f){return i+","+l+" "+(i+y)+","+l+" "+(i+y)+","+(l+h-f)+" "+(i+y-1.2*f)+","+(l+h)+" "+i+","+(l+h)}s(r,"genPoints");const c=t.append("polygon");c.attr("points",r(e.x,e.y,50,20,7)),c.attr("class","labelBox"),e.y=e.y+e.labelMargin,e.x=e.x+.5*e.labelMargin,st(t,e)},"drawLabel"),Bt=s(function(t,e,r){const c=t.append("g"),i=K();i.x=e.x,i.y=e.y,i.fill=e.fill,i.width=r.width*e.taskCount+r.diagramMarginX*(e.taskCount-1),i.height=r.height,i.class="journey-section section-type-"+e.num,i.rx=3,i.ry=3,X(c,i),rt(r)(e.text,c,i.x,i.y,i.width,i.height,{class:"journey-section section-type-"+e.num},r,e.colour)},"drawSection"),at=-1,Lt=s(function(t,e,r){const c=e.x+r.width/2,i=t.append("g");at++,i.append("line").attr("id","task"+at).attr("x1",c).attr("y1",e.y).attr("x2",c).attr("y2",450).attr("class","task-line").attr("stroke-width","1px").attr("stroke-dasharray","4 2").attr("stroke","#666"),Ft(i,{cx:c,cy:300+30*(5-e.score),score:e.score});const l=K();l.x=e.x,l.y=e.y,l.fill=e.fill,l.width=r.width,l.height=r.height,l.class="task task-type-"+e.num,l.rx=3,l.ry=3,X(i,l);let y=e.x+14;e.people.forEach(h=>{const f=e.actors[h].color,g={cx:y,cy:e.y,r:7,fill:f,stroke:"#000",title:h,pos:e.actors[h].position};it(i,g),y+=10}),rt(r)(e.task,i,l.x,l.y,l.width,l.height,{class:"task"},r,e.colour)},"drawTask"),Rt=s(function(t,e){pt(t,e)},"drawBackgroundRect"),rt=function(){function t(i,l,y,h,f,g,m,n){c(l.append("text").attr("x",y+f/2).attr("y",h+g/2+5).style("font-color",n).style("text-anchor","middle").text(i),m)}function e(i,l,y,h,f,g,m,n,u){const{taskFontSize:a,taskFontFamily:p}=n,d=i.split(/<br\s*\/?>/gi);for(let o=0;o<d.length;o++){const $=o*a-a*(d.length-1)/2,x=l.append("text").attr("x",y+f/2).attr("y",h).attr("fill",u).style("text-anchor","middle").style("font-size",a).style("font-family",p);x.append("tspan").attr("x",y+f/2).attr("dy",$).text(d[o]),x.attr("y",h+g/2).attr("dominant-baseline","central").attr("alignment-baseline","central"),c(x,m)}}function r(i,l,y,h,f,g,m,n){const u=l.append("switch"),a=u.append("foreignObject").attr("x",y).attr("y",h).attr("width",f).attr("height",g).attr("position","fixed").append("xhtml:div").style("display","table").style("height","100%").style("width","100%");a.append("div").attr("class","label").style("display","table-cell").style("text-align","center").style("vertical-align","middle").text(i),e(i,u,y,h,f,g,m,n),c(a,m)}function c(i,l){for(const y in l)y in l&&i.attr(y,l[y])}return s(t,"byText"),s(e,"byTspan"),s(r,"byFo"),s(c,"_setTextAttrs"),function(i){return i.textPlacement==="fo"?r:i.textPlacement==="old"?t:e}}(),R={drawRect:X,drawCircle:it,drawSection:Bt,drawText:st,drawLabel:Vt,drawTask:Lt,drawBackgroundRect:Rt,initGraphics:s(function(t){t.append("defs").append("marker").attr("id","arrowhead").attr("refX",5).attr("refY",2).attr("markerWidth",6).attr("markerHeight",4).attr("orient","auto").append("path").attr("d","M 0,0 V 4 L6,2 Z")},"initGraphics")},Ot=s(function(t){Object.keys(t).forEach(function(e){M[e]=t[e]})},"setConf"),S={},N=0;function ot(t){const e=F().journey,r=e.maxLabelWidth;N=0;let c=60;Object.keys(S).forEach(i=>{const l=S[i].color,y={cx:20,cy:c,r:7,fill:l,stroke:"#000",pos:S[i].position};R.drawCircle(t,y);let h=t.append("text").attr("visibility","hidden").text(i);const f=h.node().getBoundingClientRect().width;h.remove();let g=[];if(f<=r)g=[i];else{const m=i.split(" ");let n="";h=t.append("text").attr("visibility","hidden"),m.forEach(u=>{const a=n?`${n} ${u}`:u;if(h.text(a),h.node().getBoundingClientRect().width>r){if(n&&g.push(n),n=u,h.text(u),h.node().getBoundingClientRect().width>r){let p="";for(const d of u)p+=d,h.text(p+"-"),h.node().getBoundingClientRect().width>r&&(g.push(p.slice(0,-1)+"-"),p=d);n=p}}else n=a}),n&&g.push(n),h.remove()}g.forEach((m,n)=>{const u={x:40,y:c+7+20*n,fill:"#666",text:m,textMargin:e.boxTextMargin??5},a=R.drawText(t,u).node().getBoundingClientRect().width;a>N&&a>e.leftMargin-a&&(N=a)}),c+=Math.max(20,20*g.length)})}s(ot,"drawActorLegend");var M=F().journey,C=0,Dt=s(function(t,e,r,c){const i=F(),l=i.journey.titleColor,y=i.journey.titleFontSize,h=i.journey.titleFontFamily,f=i.securityLevel;let g;f==="sandbox"&&(g=Q("#i"+e));const m=Q(f==="sandbox"?g.nodes()[0].contentDocument.body:"body");v.init();const n=m.select("#"+e);R.initGraphics(n);const u=c.db.getTasks(),a=c.db.getDiagramTitle(),p=c.db.getActors();for(const P in S)delete S[P];let d=0;p.forEach(P=>{S[P]={color:M.actorColours[d%M.actorColours.length],position:d},d++}),ot(n),C=M.leftMargin+N,v.insert(0,0,C,50*Object.keys(S).length),Nt(n,u,0);const o=v.getBounds();a&&n.append("text").text(a).attr("x",C).attr("font-size",y).attr("font-weight","bold").attr("y",25).attr("fill",l).attr("font-family",h);const $=o.stopy-o.starty+2*M.diagramMarginY,x=C+o.stopx+2*M.diagramMarginX;wt(n,$,x,M.useMaxWidth),n.append("line").attr("x1",C).attr("y1",4*M.height).attr("x2",x-C-4).attr("y2",4*M.height).attr("stroke-width",4).attr("stroke","black").attr("marker-end","url(#arrowhead)");const E=a?70:0;n.attr("viewBox",`${o.startx} -25 ${x} ${$+E}`),n.attr("preserveAspectRatio","xMinYMin meet"),n.attr("height",$+E+25)},"draw"),v={data:{startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},verticalPos:0,sequenceItems:[],init:s(function(){this.sequenceItems=[],this.data={startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},this.verticalPos=0},"init"),updateVal:s(function(t,e,r,c){t[e]===void 0?t[e]=r:t[e]=c(r,t[e])},"updateVal"),updateBounds:s(function(t,e,r,c){const i=F().journey,l=this;let y=0;function h(f){return s(function(g){y++;const m=l.sequenceItems.length-y+1;l.updateVal(g,"starty",e-m*i.boxMargin,Math.min),l.updateVal(g,"stopy",c+m*i.boxMargin,Math.max),l.updateVal(v.data,"startx",t-m*i.boxMargin,Math.min),l.updateVal(v.data,"stopx",r+m*i.boxMargin,Math.max),f!=="activation"&&(l.updateVal(g,"startx",t-m*i.boxMargin,Math.min),l.updateVal(g,"stopx",r+m*i.boxMargin,Math.max),l.updateVal(v.data,"starty",e-m*i.boxMargin,Math.min),l.updateVal(v.data,"stopy",c+m*i.boxMargin,Math.max))},"updateItemBounds")}s(h,"updateFn"),this.sequenceItems.forEach(h())},"updateBounds"),insert:s(function(t,e,r,c){const i=Math.min(t,r),l=Math.max(t,r),y=Math.min(e,c),h=Math.max(e,c);this.updateVal(v.data,"startx",i,Math.min),this.updateVal(v.data,"starty",y,Math.min),this.updateVal(v.data,"stopx",l,Math.max),this.updateVal(v.data,"stopy",h,Math.max),this.updateBounds(i,y,l,h)},"insert"),bumpVerticalPos:s(function(t){this.verticalPos=this.verticalPos+t,this.data.stopy=this.verticalPos},"bumpVerticalPos"),getVerticalPos:s(function(){return this.verticalPos},"getVerticalPos"),getBounds:s(function(){return this.data},"getBounds")},G=M.sectionFills,lt=M.sectionColours,Nt=s(function(t,e,r){const c=F().journey;let i="";const l=r+(2*c.height+c.diagramMarginY);let y=0,h="#CCC",f="black",g=0;for(const[m,n]of e.entries()){if(i!==n.section){h=G[y%G.length],g=y%G.length,f=lt[y%lt.length];let a=0;const p=n.section;for(let o=m;o<e.length&&e[o].section==p;o++)a+=1;const d={x:m*c.taskMargin+m*c.width+C,y:50,text:n.section,fill:h,num:g,colour:f,taskCount:a};R.drawSection(t,d,c),i=n.section,y++}const u=n.people.reduce((a,p)=>(S[p]&&(a[p]=S[p]),a),{});n.x=m*c.taskMargin+m*c.width+C,n.y=l,n.width=c.diagramMarginX,n.height=c.diagramMarginY,n.colour=f,n.fill=h,n.num=g,n.actors=u,R.drawTask(t,n,c),v.insert(n.x,n.y,n.x+n.width+c.taskMargin,450)}},"drawTasks"),ct={setConf:Ot,draw:Dt},zt={parser:$t,db:nt,renderer:ct,styles:jt,init:s(t=>{ct.setConf(t.journey),nt.clear()},"init")};export{zt as diagram};
