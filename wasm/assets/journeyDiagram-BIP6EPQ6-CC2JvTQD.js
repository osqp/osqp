import"./preload-helper-DImqtvgl.js";import"./timer-m_pEB4Lb.js";import{u as tt}from"./src--EmJf_Ct.js";import"./path-Gc-fQZTe.js";import"./math-BsaXoFIn.js";import{t as et}from"./arc-De5rudRM.js";import{n as r}from"./src-CWnjMQt8.js";import{B as gt,C as mt,U as xt,_ as kt,a as _t,b as j,c as bt,v as vt,z as $t}from"./chunk-ABZYJK2D-eZsthrBr.js";import"./dist-D2dAPhhG.js";import{t as wt}from"./chunk-FMBD7UC4-CuAAQZLl.js";import{a as Mt,i as Tt,o as it,t as Ct}from"./chunk-TZMSLE5B-DdgcuCwZ.js";var X=(function(){var t=r(function(i,a,n,u){for(n||(n={}),u=i.length;u--;n[i[u]]=a);return n},"o"),e=[6,8,10,11,12,14,16,17,18],l=[1,9],c=[1,10],s=[1,11],h=[1,12],d=[1,13],p=[1,14],g={trace:r(function(){},"trace"),yy:{},symbols_:{error:2,start:3,journey:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NEWLINE:10,title:11,acc_title:12,acc_title_value:13,acc_descr:14,acc_descr_value:15,acc_descr_multiline_value:16,section:17,taskName:18,taskData:19,$accept:0,$end:1},terminals_:{2:"error",4:"journey",6:"EOF",8:"SPACE",10:"NEWLINE",11:"title",12:"acc_title",13:"acc_title_value",14:"acc_descr",15:"acc_descr_value",16:"acc_descr_multiline_value",17:"section",18:"taskName",19:"taskData"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,2]],performAction:r(function(i,a,n,u,y,o,k){var m=o.length-1;switch(y){case 1:return o[m-1];case 2:this.$=[];break;case 3:o[m-1].push(o[m]),this.$=o[m-1];break;case 4:case 5:this.$=o[m];break;case 6:case 7:this.$=[];break;case 8:u.setDiagramTitle(o[m].substr(6)),this.$=o[m].substr(6);break;case 9:this.$=o[m].trim(),u.setAccTitle(this.$);break;case 10:case 11:this.$=o[m].trim(),u.setAccDescription(this.$);break;case 12:u.addSection(o[m].substr(8)),this.$=o[m].substr(8);break;case 13:u.addTask(o[m-1],o[m]),this.$="task";break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:l,12:c,14:s,16:h,17:d,18:p},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:15,11:l,12:c,14:s,16:h,17:d,18:p},t(e,[2,5]),t(e,[2,6]),t(e,[2,8]),{13:[1,16]},{15:[1,17]},t(e,[2,11]),t(e,[2,12]),{19:[1,18]},t(e,[2,4]),t(e,[2,9]),t(e,[2,10]),t(e,[2,13])],defaultActions:{},parseError:r(function(i,a){if(a.recoverable)this.trace(i);else{var n=Error(i);throw n.hash=a,n}},"parseError"),parse:r(function(i){var a=this,n=[0],u=[],y=[null],o=[],k=this.table,m="",b=0,B=0,A=0,pt=2,Z=1,yt=o.slice.call(arguments,1),x=Object.create(this.lexer),I={yy:{}};for(var D in this.yy)Object.prototype.hasOwnProperty.call(this.yy,D)&&(I.yy[D]=this.yy[D]);x.setInput(i,I.yy),I.yy.lexer=x,I.yy.parser=this,x.yylloc===void 0&&(x.yylloc={});var Y=x.yylloc;o.push(Y);var dt=x.options&&x.options.ranges;typeof I.yy.parseError=="function"?this.parseError=I.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ft($){n.length-=2*$,y.length-=$,o.length-=$}r(ft,"popStack");function J(){var $=u.pop()||x.lex()||Z;return typeof $!="number"&&($ instanceof Array&&(u=$,$=u.pop()),$=a.symbols_[$]||$),$}r(J,"lex");for(var _,W,S,v,q,P={},N,T,K,O;;){if(S=n[n.length-1],this.defaultActions[S]?v=this.defaultActions[S]:(_??(_=J()),v=k[S]&&k[S][_]),v===void 0||!v.length||!v[0]){var Q="";for(N in O=[],k[S])this.terminals_[N]&&N>pt&&O.push("'"+this.terminals_[N]+"'");Q=x.showPosition?"Parse error on line "+(b+1)+`:
`+x.showPosition()+`
Expecting `+O.join(", ")+", got '"+(this.terminals_[_]||_)+"'":"Parse error on line "+(b+1)+": Unexpected "+(_==Z?"end of input":"'"+(this.terminals_[_]||_)+"'"),this.parseError(Q,{text:x.match,token:this.terminals_[_]||_,line:x.yylineno,loc:Y,expected:O})}if(v[0]instanceof Array&&v.length>1)throw Error("Parse Error: multiple actions possible at state: "+S+", token: "+_);switch(v[0]){case 1:n.push(_),y.push(x.yytext),o.push(x.yylloc),n.push(v[1]),_=null,W?(_=W,W=null):(B=x.yyleng,m=x.yytext,b=x.yylineno,Y=x.yylloc,A>0&&A--);break;case 2:if(T=this.productions_[v[1]][1],P.$=y[y.length-T],P._$={first_line:o[o.length-(T||1)].first_line,last_line:o[o.length-1].last_line,first_column:o[o.length-(T||1)].first_column,last_column:o[o.length-1].last_column},dt&&(P._$.range=[o[o.length-(T||1)].range[0],o[o.length-1].range[1]]),q=this.performAction.apply(P,[m,B,b,I.yy,v[1],y,o].concat(yt)),q!==void 0)return q;T&&(n=n.slice(0,-1*T*2),y=y.slice(0,-1*T),o=o.slice(0,-1*T)),n.push(this.productions_[v[1]][0]),y.push(P.$),o.push(P._$),K=k[n[n.length-2]][n[n.length-1]],n.push(K);break;case 3:return!0}}return!0},"parse")};g.lexer=(function(){return{EOF:1,parseError:r(function(i,a){if(this.yy.parser)this.yy.parser.parseError(i,a);else throw Error(i)},"parseError"),setInput:r(function(i,a){return this.yy=a||this.yy||{},this._input=i,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:r(function(){var i=this._input[0];return this.yytext+=i,this.yyleng++,this.offset++,this.match+=i,this.matched+=i,i.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),i},"input"),unput:r(function(i){var a=i.length,n=i.split(/(?:\r\n?|\n)/g);this._input=i+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-a),this.offset-=a;var u=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),n.length-1&&(this.yylineno-=n.length-1);var y=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:n?(n.length===u.length?this.yylloc.first_column:0)+u[u.length-n.length].length-n[0].length:this.yylloc.first_column-a},this.options.ranges&&(this.yylloc.range=[y[0],y[0]+this.yyleng-a]),this.yyleng=this.yytext.length,this},"unput"),more:r(function(){return this._more=!0,this},"more"),reject:r(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:r(function(i){this.unput(this.match.slice(i))},"less"),pastInput:r(function(){var i=this.matched.substr(0,this.matched.length-this.match.length);return(i.length>20?"...":"")+i.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:r(function(){var i=this.match;return i.length<20&&(i+=this._input.substr(0,20-i.length)),(i.substr(0,20)+(i.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:r(function(){var i=this.pastInput(),a=Array(i.length+1).join("-");return i+this.upcomingInput()+`
`+a+"^"},"showPosition"),test_match:r(function(i,a){var n,u,y;if(this.options.backtrack_lexer&&(y={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(y.yylloc.range=this.yylloc.range.slice(0))),u=i[0].match(/(?:\r\n?|\n).*/g),u&&(this.yylineno+=u.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:u?u[u.length-1].length-u[u.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+i[0].length},this.yytext+=i[0],this.match+=i[0],this.matches=i,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(i[0].length),this.matched+=i[0],n=this.performAction.call(this,this.yy,this,a,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),n)return n;if(this._backtrack){for(var o in y)this[o]=y[o];return!1}return!1},"test_match"),next:r(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var i,a,n,u;this._more||(this.yytext="",this.match="");for(var y=this._currentRules(),o=0;o<y.length;o++)if(n=this._input.match(this.rules[y[o]]),n&&(!a||n[0].length>a[0].length)){if(a=n,u=o,this.options.backtrack_lexer){if(i=this.test_match(n,y[o]),i!==!1)return i;if(this._backtrack){a=!1;continue}else return!1}else if(!this.options.flex)break}return a?(i=this.test_match(a,y[u]),i===!1?!1:i):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:r(function(){return this.next()||this.lex()},"lex"),begin:r(function(i){this.conditionStack.push(i)},"begin"),popState:r(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:r(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:r(function(i){return i=this.conditionStack.length-1-Math.abs(i||0),i>=0?this.conditionStack[i]:"INITIAL"},"topState"),pushState:r(function(i){this.begin(i)},"pushState"),stateStackSize:r(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:r(function(i,a,n,u){switch(n){case 0:break;case 1:break;case 2:return 10;case 3:break;case 4:break;case 5:return 4;case 6:return 11;case 7:return this.begin("acc_title"),12;case 8:return this.popState(),"acc_title_value";case 9:return this.begin("acc_descr"),14;case 10:return this.popState(),"acc_descr_value";case 11:this.begin("acc_descr_multiline");break;case 12:this.popState();break;case 13:return"acc_descr_multiline_value";case 14:return 17;case 15:return 18;case 16:return 19;case 17:return":";case 18:return 6;case 19:return"INVALID"}},"anonymous"),rules:[/^(?:%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:#[^\n]*)/i,/^(?:journey\b)/i,/^(?:title\s[^#\n;]+)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:section\s[^#:\n;]+)/i,/^(?:[^#:\n;]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[12,13],inclusive:!1},acc_descr:{rules:[10],inclusive:!1},acc_title:{rules:[8],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,9,11,14,15,16,17,18,19],inclusive:!0}}}})();function f(){this.yy={}}return r(f,"Parser"),f.prototype=g,g.Parser=f,new f})();X.parser=X;var Et=X,F="",G=[],V=[],L=[],It=r(function(){G.length=0,V.length=0,F="",L.length=0,_t()},"clear"),St=r(function(t){F=t,G.push(t)},"addSection"),At=r(function(){return G},"getSections"),Pt=r(function(){let t=nt(),e=0;for(;!t&&e<100;)t=nt(),e++;return V.push(...L),V},"getTasks"),jt=r(function(){let t=[];return V.forEach(e=>{e.people&&t.push(...e.people)}),[...new Set(t)].sort()},"updateActors"),Ft=r(function(t,e){let l=e.substr(1).split(":"),c=0,s=[];l.length===1?(c=Number(l[0]),s=[]):(c=Number(l[0]),s=l[1].split(","));let h=s.map(p=>p.trim()),d={section:F,type:F,people:h,task:t,score:c};L.push(d)},"addTask"),Bt=r(function(t){let e={section:F,type:F,description:t,task:t,classes:[]};V.push(e)},"addTaskOrg"),nt=r(function(){let t=r(function(l){return L[l].processed},"compileTask"),e=!0;for(let[l,c]of L.entries())t(l),e&&(e=c.processed);return e},"compileTasks"),st={getConfig:r(()=>j().journey,"getConfig"),clear:It,setDiagramTitle:xt,getDiagramTitle:mt,setAccTitle:gt,getAccTitle:vt,setAccDescription:$t,getAccDescription:kt,addSection:St,getSections:At,getTasks:Pt,addTask:Ft,addTaskOrg:Bt,getActors:r(function(){return jt()},"getActors")},Vt=r(t=>`.label {
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
  ${wt()}
`,"getStyles"),U=r(function(t,e){return Tt(t,e)},"drawRect"),Lt=r(function(t,e){let l=t.append("circle").attr("cx",e.cx).attr("cy",e.cy).attr("class","face").attr("r",15).attr("stroke-width",2).attr("overflow","visible"),c=t.append("g");c.append("circle").attr("cx",e.cx-15/3).attr("cy",e.cy-15/3).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666"),c.append("circle").attr("cx",e.cx+15/3).attr("cy",e.cy-15/3).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666");function s(p){let g=et().startAngle(Math.PI/2).endAngle(3*(Math.PI/2)).innerRadius(7.5).outerRadius(6.8181818181818175);p.append("path").attr("class","mouth").attr("d",g).attr("transform","translate("+e.cx+","+(e.cy+2)+")")}r(s,"smile");function h(p){let g=et().startAngle(3*Math.PI/2).endAngle(5*(Math.PI/2)).innerRadius(7.5).outerRadius(6.8181818181818175);p.append("path").attr("class","mouth").attr("d",g).attr("transform","translate("+e.cx+","+(e.cy+7)+")")}r(h,"sad");function d(p){p.append("line").attr("class","mouth").attr("stroke",2).attr("x1",e.cx-5).attr("y1",e.cy+7).attr("x2",e.cx+5).attr("y2",e.cy+7).attr("class","mouth").attr("stroke-width","1px").attr("stroke","#666")}return r(d,"ambivalent"),e.score>3?s(c):e.score<3?h(c):d(c),l},"drawFace"),rt=r(function(t,e){let l=t.append("circle");return l.attr("cx",e.cx),l.attr("cy",e.cy),l.attr("class","actor-"+e.pos),l.attr("fill",e.fill),l.attr("stroke",e.stroke),l.attr("r",e.r),l.class!==void 0&&l.attr("class",l.class),e.title!==void 0&&l.append("title").text(e.title),l},"drawCircle"),at=r(function(t,e){return Mt(t,e)},"drawText"),Rt=r(function(t,e){function l(s,h,d,p,g){return s+","+h+" "+(s+d)+","+h+" "+(s+d)+","+(h+p-g)+" "+(s+d-g*1.2)+","+(h+p)+" "+s+","+(h+p)}r(l,"genPoints");let c=t.append("polygon");c.attr("points",l(e.x,e.y,50,20,7)),c.attr("class","labelBox"),e.y+=e.labelMargin,e.x+=.5*e.labelMargin,at(t,e)},"drawLabel"),Nt=r(function(t,e,l){let c=t.append("g"),s=it();s.x=e.x,s.y=e.y,s.fill=e.fill,s.width=l.width*e.taskCount+l.diagramMarginX*(e.taskCount-1),s.height=l.height,s.class="journey-section section-type-"+e.num,s.rx=3,s.ry=3,U(c,s),lt(l)(e.text,c,s.x,s.y,s.width,s.height,{class:"journey-section section-type-"+e.num},l,e.colour)},"drawSection"),ot=-1,Ot=r(function(t,e,l){let c=e.x+l.width/2,s=t.append("g");ot++,s.append("line").attr("id","task"+ot).attr("x1",c).attr("y1",e.y).attr("x2",c).attr("y2",450).attr("class","task-line").attr("stroke-width","1px").attr("stroke-dasharray","4 2").attr("stroke","#666"),Lt(s,{cx:c,cy:300+(5-e.score)*30,score:e.score});let h=it();h.x=e.x,h.y=e.y,h.fill=e.fill,h.width=l.width,h.height=l.height,h.class="task task-type-"+e.num,h.rx=3,h.ry=3,U(s,h);let d=e.x+14;e.people.forEach(p=>{let g=e.actors[p].color,f={cx:d,cy:e.y,r:7,fill:g,stroke:"#000",title:p,pos:e.actors[p].position};rt(s,f),d+=10}),lt(l)(e.task,s,h.x,h.y,h.width,h.height,{class:"task"},l,e.colour)},"drawTask"),zt=r(function(t,e){Ct(t,e)},"drawBackgroundRect"),lt=(function(){function t(s,h,d,p,g,f,i,a){let n=h.append("text").attr("x",d+g/2).attr("y",p+f/2+5).style("font-color",a).style("text-anchor","middle").text(s);c(n,i)}r(t,"byText");function e(s,h,d,p,g,f,i,a,n){let{taskFontSize:u,taskFontFamily:y}=a,o=s.split(/<br\s*\/?>/gi);for(let k=0;k<o.length;k++){let m=k*u-u*(o.length-1)/2,b=h.append("text").attr("x",d+g/2).attr("y",p).attr("fill",n).style("text-anchor","middle").style("font-size",u).style("font-family",y);b.append("tspan").attr("x",d+g/2).attr("dy",m).text(o[k]),b.attr("y",p+f/2).attr("dominant-baseline","central").attr("alignment-baseline","central"),c(b,i)}}r(e,"byTspan");function l(s,h,d,p,g,f,i,a){let n=h.append("switch"),u=n.append("foreignObject").attr("x",d).attr("y",p).attr("width",g).attr("height",f).attr("position","fixed").append("xhtml:div").style("display","table").style("height","100%").style("width","100%");u.append("div").attr("class","label").style("display","table-cell").style("text-align","center").style("vertical-align","middle").text(s),e(s,n,d,p,g,f,i,a),c(u,i)}r(l,"byFo");function c(s,h){for(let d in h)d in h&&s.attr(d,h[d])}return r(c,"_setTextAttrs"),function(s){return s.textPlacement==="fo"?l:s.textPlacement==="old"?t:e}})(),R={drawRect:U,drawCircle:rt,drawSection:Nt,drawText:at,drawLabel:Rt,drawTask:Ot,drawBackgroundRect:zt,initGraphics:r(function(t){t.append("defs").append("marker").attr("id","arrowhead").attr("refX",5).attr("refY",2).attr("markerWidth",6).attr("markerHeight",4).attr("orient","auto").append("path").attr("d","M 0,0 V 4 L6,2 Z")},"initGraphics")},Dt=r(function(t){Object.keys(t).forEach(function(e){M[e]=t[e]})},"setConf"),C={},z=0;function ct(t){let e=j().journey,l=e.maxLabelWidth;z=0;let c=60;Object.keys(C).forEach(s=>{let h=C[s].color,d={cx:20,cy:c,r:7,fill:h,stroke:"#000",pos:C[s].position};R.drawCircle(t,d);let p=t.append("text").attr("visibility","hidden").text(s),g=p.node().getBoundingClientRect().width;p.remove();let f=[];if(g<=l)f=[s];else{let i=s.split(" "),a="";p=t.append("text").attr("visibility","hidden"),i.forEach(n=>{let u=a?`${a} ${n}`:n;if(p.text(u),p.node().getBoundingClientRect().width>l){if(a&&f.push(a),a=n,p.text(n),p.node().getBoundingClientRect().width>l){let y="";for(let o of n)y+=o,p.text(y+"-"),p.node().getBoundingClientRect().width>l&&(f.push(y.slice(0,-1)+"-"),y=o);a=y}}else a=u}),a&&f.push(a),p.remove()}f.forEach((i,a)=>{let n={x:40,y:c+7+a*20,fill:"#666",text:i,textMargin:e.boxTextMargin??5},u=R.drawText(t,n).node().getBoundingClientRect().width;u>z&&u>e.leftMargin-u&&(z=u)}),c+=Math.max(20,f.length*20)})}r(ct,"drawActorLegend");var M=j().journey,E=0,Yt=r(function(t,e,l,c){let s=j(),h=s.journey.titleColor,d=s.journey.titleFontSize,p=s.journey.titleFontFamily,g=s.securityLevel,f;g==="sandbox"&&(f=tt("#i"+e));let i=tt(g==="sandbox"?f.nodes()[0].contentDocument.body:"body");w.init();let a=i.select("#"+e);R.initGraphics(a);let n=c.db.getTasks(),u=c.db.getDiagramTitle(),y=c.db.getActors();for(let A in C)delete C[A];let o=0;y.forEach(A=>{C[A]={color:M.actorColours[o%M.actorColours.length],position:o},o++}),ct(a),E=M.leftMargin+z,w.insert(0,0,E,Object.keys(C).length*50),Wt(a,n,0);let k=w.getBounds();u&&a.append("text").text(u).attr("x",E).attr("font-size",d).attr("font-weight","bold").attr("y",25).attr("fill",h).attr("font-family",p);let m=k.stopy-k.starty+2*M.diagramMarginY,b=E+k.stopx+2*M.diagramMarginX;bt(a,m,b,M.useMaxWidth),a.append("line").attr("x1",E).attr("y1",M.height*4).attr("x2",b-E-4).attr("y2",M.height*4).attr("stroke-width",4).attr("stroke","black").attr("marker-end","url(#arrowhead)");let B=u?70:0;a.attr("viewBox",`${k.startx} -25 ${b} ${m+B}`),a.attr("preserveAspectRatio","xMinYMin meet"),a.attr("height",m+B+25)},"draw"),w={data:{startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},verticalPos:0,sequenceItems:[],init:r(function(){this.sequenceItems=[],this.data={startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},this.verticalPos=0},"init"),updateVal:r(function(t,e,l,c){t[e]===void 0?t[e]=l:t[e]=c(l,t[e])},"updateVal"),updateBounds:r(function(t,e,l,c){let s=j().journey,h=this,d=0;function p(g){return r(function(f){d++;let i=h.sequenceItems.length-d+1;h.updateVal(f,"starty",e-i*s.boxMargin,Math.min),h.updateVal(f,"stopy",c+i*s.boxMargin,Math.max),h.updateVal(w.data,"startx",t-i*s.boxMargin,Math.min),h.updateVal(w.data,"stopx",l+i*s.boxMargin,Math.max),g!=="activation"&&(h.updateVal(f,"startx",t-i*s.boxMargin,Math.min),h.updateVal(f,"stopx",l+i*s.boxMargin,Math.max),h.updateVal(w.data,"starty",e-i*s.boxMargin,Math.min),h.updateVal(w.data,"stopy",c+i*s.boxMargin,Math.max))},"updateItemBounds")}r(p,"updateFn"),this.sequenceItems.forEach(p())},"updateBounds"),insert:r(function(t,e,l,c){let s=Math.min(t,l),h=Math.max(t,l),d=Math.min(e,c),p=Math.max(e,c);this.updateVal(w.data,"startx",s,Math.min),this.updateVal(w.data,"starty",d,Math.min),this.updateVal(w.data,"stopx",h,Math.max),this.updateVal(w.data,"stopy",p,Math.max),this.updateBounds(s,d,h,p)},"insert"),bumpVerticalPos:r(function(t){this.verticalPos+=t,this.data.stopy=this.verticalPos},"bumpVerticalPos"),getVerticalPos:r(function(){return this.verticalPos},"getVerticalPos"),getBounds:r(function(){return this.data},"getBounds")},H=M.sectionFills,ht=M.sectionColours,Wt=r(function(t,e,l){let c=j().journey,s="",h=c.height*2+c.diagramMarginY,d=l+h,p=0,g="#CCC",f="black",i=0;for(let[a,n]of e.entries()){if(s!==n.section){g=H[p%H.length],i=p%H.length,f=ht[p%ht.length];let y=0,o=n.section;for(let m=a;m<e.length&&e[m].section==o;m++)y+=1;let k={x:a*c.taskMargin+a*c.width+E,y:50,text:n.section,fill:g,num:i,colour:f,taskCount:y};R.drawSection(t,k,c),s=n.section,p++}let u=n.people.reduce((y,o)=>(C[o]&&(y[o]=C[o]),y),{});n.x=a*c.taskMargin+a*c.width+E,n.y=d,n.width=c.diagramMarginX,n.height=c.diagramMarginY,n.colour=f,n.fill=g,n.num=i,n.actors=u,R.drawTask(t,n,c),w.insert(n.x,n.y,n.x+n.width+c.taskMargin,450)}},"drawTasks"),ut={setConf:Dt,draw:Yt},qt={parser:Et,db:st,renderer:ut,styles:Vt,init:r(t=>{ut.setConf(t.journey),st.clear()},"init")};export{qt as diagram};
