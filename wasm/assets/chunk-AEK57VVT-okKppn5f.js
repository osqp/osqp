var Kt=Object.defineProperty;var qt=(e,t,s)=>t in e?Kt(e,t,{enumerable:!0,configurable:!0,writable:!0,value:s}):e[t]=s;var k=(e,t,s)=>qt(e,typeof t!="symbol"?t+"":t,s);var M;import{g as Qt,s as Zt}from"./chunk-RZ5BOZE2-CW6KiigC.js";import{_ as l,l as E,c as $,r as te,u as ee,R as se,j as U,y as ie,a as ne,b as re,g as ae,s as oe,p as ce,q as le}from"./mermaid-Da56-Xo5.js";var mt=function(){var e=l(function(r,y,u,f){for(u=u||{},f=r.length;f--;u[r[f]]=y);return u},"o"),t=[1,2],s=[1,3],a=[1,4],o=[2,4],i=[1,9],g=[1,11],d=[1,16],c=[1,17],S=[1,18],b=[1,19],v=[1,32],F=[1,20],P=[1,21],L=[1,22],p=[1,23],I=[1,24],A=[1,26],Y=[1,27],G=[1,28],N=[1,29],R=[1,30],Q=[1,31],Z=[1,34],tt=[1,35],et=[1,36],st=[1,37],V=[1,33],m=[1,4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,42,45,48,49,50,51,54],it=[1,4,5,14,15,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,42,45,48,49,50,51,54],Et=[4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,42,45,48,49,50,51,54],pt={trace:l(function(){},"trace"),yy:{},symbols_:{error:2,start:3,SPACE:4,NL:5,SD:6,document:7,line:8,statement:9,classDefStatement:10,styleStatement:11,cssClassStatement:12,idStatement:13,DESCR:14,"-->":15,HIDE_EMPTY:16,scale:17,WIDTH:18,COMPOSIT_STATE:19,STRUCT_START:20,STRUCT_STOP:21,STATE_DESCR:22,AS:23,ID:24,FORK:25,JOIN:26,CHOICE:27,CONCURRENT:28,note:29,notePosition:30,NOTE_TEXT:31,direction:32,acc_title:33,acc_title_value:34,acc_descr:35,acc_descr_value:36,acc_descr_multiline_value:37,classDef:38,CLASSDEF_ID:39,CLASSDEF_STYLEOPTS:40,DEFAULT:41,style:42,STYLE_IDS:43,STYLEDEF_STYLEOPTS:44,class:45,CLASSENTITY_IDS:46,STYLECLASS:47,direction_tb:48,direction_bt:49,direction_rl:50,direction_lr:51,eol:52,";":53,EDGE_STATE:54,STYLE_SEPARATOR:55,left_of:56,right_of:57,$accept:0,$end:1},terminals_:{2:"error",4:"SPACE",5:"NL",6:"SD",14:"DESCR",15:"-->",16:"HIDE_EMPTY",17:"scale",18:"WIDTH",19:"COMPOSIT_STATE",20:"STRUCT_START",21:"STRUCT_STOP",22:"STATE_DESCR",23:"AS",24:"ID",25:"FORK",26:"JOIN",27:"CHOICE",28:"CONCURRENT",29:"note",31:"NOTE_TEXT",33:"acc_title",34:"acc_title_value",35:"acc_descr",36:"acc_descr_value",37:"acc_descr_multiline_value",38:"classDef",39:"CLASSDEF_ID",40:"CLASSDEF_STYLEOPTS",41:"DEFAULT",42:"style",43:"STYLE_IDS",44:"STYLEDEF_STYLEOPTS",45:"class",46:"CLASSENTITY_IDS",47:"STYLECLASS",48:"direction_tb",49:"direction_bt",50:"direction_rl",51:"direction_lr",53:";",54:"EDGE_STATE",55:"STYLE_SEPARATOR",56:"left_of",57:"right_of"},productions_:[0,[3,2],[3,2],[3,2],[7,0],[7,2],[8,2],[8,1],[8,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,3],[9,4],[9,1],[9,2],[9,1],[9,4],[9,3],[9,6],[9,1],[9,1],[9,1],[9,1],[9,4],[9,4],[9,1],[9,2],[9,2],[9,1],[10,3],[10,3],[11,3],[12,3],[32,1],[32,1],[32,1],[32,1],[52,1],[52,1],[13,1],[13,1],[13,3],[13,3],[30,1],[30,1]],performAction:l(function(r,y,u,f,_,n,W){var h=n.length-1;switch(_){case 3:return f.setRootDoc(n[h]),n[h];case 4:this.$=[];break;case 5:n[h]!="nl"&&(n[h-1].push(n[h]),this.$=n[h-1]);break;case 6:case 7:case 12:this.$=n[h];break;case 8:this.$="nl";break;case 13:const T=n[h-1];T.description=f.trimColon(n[h]),this.$=T;break;case 14:this.$={stmt:"relation",state1:n[h-2],state2:n[h]};break;case 15:const w=f.trimColon(n[h]);this.$={stmt:"relation",state1:n[h-3],state2:n[h-1],description:w};break;case 19:this.$={stmt:"state",id:n[h-3],type:"default",description:"",doc:n[h-1]};break;case 20:var j=n[h],X=n[h-2].trim();if(n[h].match(":")){var rt=n[h].split(":");j=rt[0],X=[X,rt[1]]}this.$={stmt:"state",id:j,type:"default",description:X};break;case 21:this.$={stmt:"state",id:n[h-3],type:"default",description:n[h-5],doc:n[h-1]};break;case 22:this.$={stmt:"state",id:n[h],type:"fork"};break;case 23:this.$={stmt:"state",id:n[h],type:"join"};break;case 24:this.$={stmt:"state",id:n[h],type:"choice"};break;case 25:this.$={stmt:"state",id:f.getDividerId(),type:"divider"};break;case 26:this.$={stmt:"state",id:n[h-1].trim(),note:{position:n[h-2].trim(),text:n[h].trim()}};break;case 29:this.$=n[h].trim(),f.setAccTitle(this.$);break;case 30:case 31:this.$=n[h].trim(),f.setAccDescription(this.$);break;case 32:case 33:this.$={stmt:"classDef",id:n[h-1].trim(),classes:n[h].trim()};break;case 34:this.$={stmt:"style",id:n[h-1].trim(),styleClass:n[h].trim()};break;case 35:this.$={stmt:"applyClass",id:n[h-1].trim(),styleClass:n[h].trim()};break;case 36:f.setDirection("TB"),this.$={stmt:"dir",value:"TB"};break;case 37:f.setDirection("BT"),this.$={stmt:"dir",value:"BT"};break;case 38:f.setDirection("RL"),this.$={stmt:"dir",value:"RL"};break;case 39:f.setDirection("LR"),this.$={stmt:"dir",value:"LR"};break;case 42:case 43:this.$={stmt:"state",id:n[h].trim(),type:"default",description:""};break;case 44:case 45:this.$={stmt:"state",id:n[h-2].trim(),classes:[n[h].trim()],type:"default",description:""}}},"anonymous"),table:[{3:1,4:t,5:s,6:a},{1:[3]},{3:5,4:t,5:s,6:a},{3:6,4:t,5:s,6:a},e([1,4,5,16,17,19,22,24,25,26,27,28,29,33,35,37,38,42,45,48,49,50,51,54],o,{7:7}),{1:[2,1]},{1:[2,2]},{1:[2,3],4:i,5:g,8:8,9:10,10:12,11:13,12:14,13:15,16:d,17:c,19:S,22:b,24:v,25:F,26:P,27:L,28:p,29:I,32:25,33:A,35:Y,37:G,38:N,42:R,45:Q,48:Z,49:tt,50:et,51:st,54:V},e(m,[2,5]),{9:38,10:12,11:13,12:14,13:15,16:d,17:c,19:S,22:b,24:v,25:F,26:P,27:L,28:p,29:I,32:25,33:A,35:Y,37:G,38:N,42:R,45:Q,48:Z,49:tt,50:et,51:st,54:V},e(m,[2,7]),e(m,[2,8]),e(m,[2,9]),e(m,[2,10]),e(m,[2,11]),e(m,[2,12],{14:[1,39],15:[1,40]}),e(m,[2,16]),{18:[1,41]},e(m,[2,18],{20:[1,42]}),{23:[1,43]},e(m,[2,22]),e(m,[2,23]),e(m,[2,24]),e(m,[2,25]),{30:44,31:[1,45],56:[1,46],57:[1,47]},e(m,[2,28]),{34:[1,48]},{36:[1,49]},e(m,[2,31]),{39:[1,50],41:[1,51]},{43:[1,52]},{46:[1,53]},e(it,[2,42],{55:[1,54]}),e(it,[2,43],{55:[1,55]}),e(m,[2,36]),e(m,[2,37]),e(m,[2,38]),e(m,[2,39]),e(m,[2,6]),e(m,[2,13]),{13:56,24:v,54:V},e(m,[2,17]),e(Et,o,{7:57}),{24:[1,58]},{24:[1,59]},{23:[1,60]},{24:[2,46]},{24:[2,47]},e(m,[2,29]),e(m,[2,30]),{40:[1,61]},{40:[1,62]},{44:[1,63]},{47:[1,64]},{24:[1,65]},{24:[1,66]},e(m,[2,14],{14:[1,67]}),{4:i,5:g,8:8,9:10,10:12,11:13,12:14,13:15,16:d,17:c,19:S,21:[1,68],22:b,24:v,25:F,26:P,27:L,28:p,29:I,32:25,33:A,35:Y,37:G,38:N,42:R,45:Q,48:Z,49:tt,50:et,51:st,54:V},e(m,[2,20],{20:[1,69]}),{31:[1,70]},{24:[1,71]},e(m,[2,32]),e(m,[2,33]),e(m,[2,34]),e(m,[2,35]),e(it,[2,44]),e(it,[2,45]),e(m,[2,15]),e(m,[2,19]),e(Et,o,{7:72}),e(m,[2,26]),e(m,[2,27]),{4:i,5:g,8:8,9:10,10:12,11:13,12:14,13:15,16:d,17:c,19:S,21:[1,73],22:b,24:v,25:F,26:P,27:L,28:p,29:I,32:25,33:A,35:Y,37:G,38:N,42:R,45:Q,48:Z,49:tt,50:et,51:st,54:V},e(m,[2,21])],defaultActions:{5:[2,1],6:[2,2],46:[2,46],47:[2,47]},parseError:l(function(r,y){if(!y.recoverable){var u=new Error(r);throw u.hash=y,u}this.trace(r)},"parseError"),parse:l(function(r){var y=this,u=[0],f=[],_=[null],n=[],W=this.table,h="",j=0,X=0,rt=n.slice.call(arguments,1),T=Object.create(this.lexer),w={yy:{}};for(var yt in this.yy)Object.prototype.hasOwnProperty.call(this.yy,yt)&&(w.yy[yt]=this.yy[yt]);T.setInput(r,w.yy),w.yy.lexer=T,w.yy.parser=this,T.yylloc===void 0&&(T.yylloc={});var gt=T.yylloc;n.push(gt);var Jt=T.options&&T.options.ranges;function Dt(){var C;return typeof(C=f.pop()||T.lex()||1)!="number"&&(C instanceof Array&&(C=(f=C).pop()),C=y.symbols_[C]||C),C}typeof w.yy.parseError=="function"?this.parseError=w.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError,l(function(C){u.length=u.length-2*C,_.length=_.length-C,n.length=n.length-C},"popStack"),l(Dt,"lex");for(var D,z,x,xt,at,O,Ct,ot,H={};;){if(z=u[u.length-1],this.defaultActions[z]?x=this.defaultActions[z]:(D==null&&(D=Dt()),x=W[z]&&W[z][D]),x===void 0||!x.length||!x[0]){var $t="";for(at in ot=[],W[z])this.terminals_[at]&&at>2&&ot.push("'"+this.terminals_[at]+"'");$t=T.showPosition?"Parse error on line "+(j+1)+`:
`+T.showPosition()+`
Expecting `+ot.join(", ")+", got '"+(this.terminals_[D]||D)+"'":"Parse error on line "+(j+1)+": Unexpected "+(D==1?"end of input":"'"+(this.terminals_[D]||D)+"'"),this.parseError($t,{text:T.match,token:this.terminals_[D]||D,line:T.yylineno,loc:gt,expected:ot})}if(x[0]instanceof Array&&x.length>1)throw new Error("Parse Error: multiple actions possible at state: "+z+", token: "+D);switch(x[0]){case 1:u.push(D),_.push(T.yytext),n.push(T.yylloc),u.push(x[1]),D=null,X=T.yyleng,h=T.yytext,j=T.yylineno,gt=T.yylloc;break;case 2:if(O=this.productions_[x[1]][1],H.$=_[_.length-O],H._$={first_line:n[n.length-(O||1)].first_line,last_line:n[n.length-1].last_line,first_column:n[n.length-(O||1)].first_column,last_column:n[n.length-1].last_column},Jt&&(H._$.range=[n[n.length-(O||1)].range[0],n[n.length-1].range[1]]),(xt=this.performAction.apply(H,[h,X,j,w.yy,x[1],_,n].concat(rt)))!==void 0)return xt;O&&(u=u.slice(0,-1*O*2),_=_.slice(0,-1*O),n=n.slice(0,-1*O)),u.push(this.productions_[x[1]][0]),_.push(H.$),n.push(H._$),Ct=W[u[u.length-2]][u[u.length-1]],u.push(Ct);break;case 3:return!0}}return!0},"parse")},Wt=function(){return{EOF:1,parseError:l(function(r,y){if(!this.yy.parser)throw new Error(r);this.yy.parser.parseError(r,y)},"parseError"),setInput:l(function(r,y){return this.yy=y||this.yy||{},this._input=r,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:l(function(){var r=this._input[0];return this.yytext+=r,this.yyleng++,this.offset++,this.match+=r,this.matched+=r,r.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),r},"input"),unput:l(function(r){var y=r.length,u=r.split(/(?:\r\n?|\n)/g);this._input=r+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-y),this.offset-=y;var f=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),u.length-1&&(this.yylineno-=u.length-1);var _=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:u?(u.length===f.length?this.yylloc.first_column:0)+f[f.length-u.length].length-u[0].length:this.yylloc.first_column-y},this.options.ranges&&(this.yylloc.range=[_[0],_[0]+this.yyleng-y]),this.yyleng=this.yytext.length,this},"unput"),more:l(function(){return this._more=!0,this},"more"),reject:l(function(){return this.options.backtrack_lexer?(this._backtrack=!0,this):this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"reject"),less:l(function(r){this.unput(this.match.slice(r))},"less"),pastInput:l(function(){var r=this.matched.substr(0,this.matched.length-this.match.length);return(r.length>20?"...":"")+r.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:l(function(){var r=this.match;return r.length<20&&(r+=this._input.substr(0,20-r.length)),(r.substr(0,20)+(r.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:l(function(){var r=this.pastInput(),y=new Array(r.length+1).join("-");return r+this.upcomingInput()+`
`+y+"^"},"showPosition"),test_match:l(function(r,y){var u,f,_;if(this.options.backtrack_lexer&&(_={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(_.yylloc.range=this.yylloc.range.slice(0))),(f=r[0].match(/(?:\r\n?|\n).*/g))&&(this.yylineno+=f.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:f?f[f.length-1].length-f[f.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+r[0].length},this.yytext+=r[0],this.match+=r[0],this.matches=r,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(r[0].length),this.matched+=r[0],u=this.performAction.call(this,this.yy,this,y,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),u)return u;if(this._backtrack){for(var n in _)this[n]=_[n];return!1}return!1},"test_match"),next:l(function(){if(this.done)return this.EOF;var r,y,u,f;this._input||(this.done=!0),this._more||(this.yytext="",this.match="");for(var _=this._currentRules(),n=0;n<_.length;n++)if((u=this._input.match(this.rules[_[n]]))&&(!y||u[0].length>y[0].length)){if(y=u,f=n,this.options.backtrack_lexer){if((r=this.test_match(u,_[n]))!==!1)return r;if(this._backtrack){y=!1;continue}return!1}if(!this.options.flex)break}return y?(r=this.test_match(y,_[f]))!==!1&&r:this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:l(function(){var r=this.next();return r||this.lex()},"lex"),begin:l(function(r){this.conditionStack.push(r)},"begin"),popState:l(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:l(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:l(function(r){return(r=this.conditionStack.length-1-Math.abs(r||0))>=0?this.conditionStack[r]:"INITIAL"},"topState"),pushState:l(function(r){this.begin(r)},"pushState"),stateStackSize:l(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:l(function(r,y,u,f){switch(u){case 0:return 41;case 1:case 42:return 48;case 2:case 43:return 49;case 3:case 44:return 50;case 4:case 45:return 51;case 5:case 6:case 8:case 9:case 10:case 11:case 54:case 56:case 62:break;case 7:case 77:return 5;case 12:case 32:return this.pushState("SCALE"),17;case 13:case 33:return 18;case 14:case 20:case 34:case 49:case 52:this.popState();break;case 15:return this.begin("acc_title"),33;case 16:return this.popState(),"acc_title_value";case 17:return this.begin("acc_descr"),35;case 18:return this.popState(),"acc_descr_value";case 19:this.begin("acc_descr_multiline");break;case 21:return"acc_descr_multiline_value";case 22:return this.pushState("CLASSDEF"),38;case 23:return this.popState(),this.pushState("CLASSDEFID"),"DEFAULT_CLASSDEF_ID";case 24:return this.popState(),this.pushState("CLASSDEFID"),39;case 25:return this.popState(),40;case 26:return this.pushState("CLASS"),45;case 27:return this.popState(),this.pushState("CLASS_STYLE"),46;case 28:return this.popState(),47;case 29:return this.pushState("STYLE"),42;case 30:return this.popState(),this.pushState("STYLEDEF_STYLES"),43;case 31:return this.popState(),44;case 35:this.pushState("STATE");break;case 36:case 39:return this.popState(),y.yytext=y.yytext.slice(0,-8).trim(),25;case 37:case 40:return this.popState(),y.yytext=y.yytext.slice(0,-8).trim(),26;case 38:case 41:return this.popState(),y.yytext=y.yytext.slice(0,-10).trim(),27;case 46:this.pushState("STATE_STRING");break;case 47:return this.pushState("STATE_ID"),"AS";case 48:case 64:return this.popState(),"ID";case 50:return"STATE_DESCR";case 51:return 19;case 53:return this.popState(),this.pushState("struct"),20;case 55:return this.popState(),21;case 57:return this.begin("NOTE"),29;case 58:return this.popState(),this.pushState("NOTE_ID"),56;case 59:return this.popState(),this.pushState("NOTE_ID"),57;case 60:this.popState(),this.pushState("FLOATING_NOTE");break;case 61:return this.popState(),this.pushState("FLOATING_NOTE_ID"),"AS";case 63:return"NOTE_TEXT";case 65:return this.popState(),this.pushState("NOTE_TEXT"),24;case 66:return this.popState(),y.yytext=y.yytext.substr(2).trim(),31;case 67:return this.popState(),y.yytext=y.yytext.slice(0,-8).trim(),31;case 68:case 69:return 6;case 70:return 16;case 71:return 54;case 72:return 24;case 73:return y.yytext=y.yytext.trim(),14;case 74:return 15;case 75:return 28;case 76:return 55;case 78:return"INVALID"}},"anonymous"),rules:[/^(?:default\b)/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:[\s]+)/i,/^(?:((?!\n)\s)+)/i,/^(?:#[^\n]*)/i,/^(?:%[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:classDef\s+)/i,/^(?:DEFAULT\s+)/i,/^(?:\w+\s+)/i,/^(?:[^\n]*)/i,/^(?:class\s+)/i,/^(?:(\w+)+((,\s*\w+)*))/i,/^(?:[^\n]*)/i,/^(?:style\s+)/i,/^(?:[\w,]+\s+)/i,/^(?:[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:state\s+)/i,/^(?:.*<<fork>>)/i,/^(?:.*<<join>>)/i,/^(?:.*<<choice>>)/i,/^(?:.*\[\[fork\]\])/i,/^(?:.*\[\[join\]\])/i,/^(?:.*\[\[choice\]\])/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:["])/i,/^(?:\s*as\s+)/i,/^(?:[^\n\{]*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n\s\{]+)/i,/^(?:\n)/i,/^(?:\{)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:\})/i,/^(?:[\n])/i,/^(?:note\s+)/i,/^(?:left of\b)/i,/^(?:right of\b)/i,/^(?:")/i,/^(?:\s*as\s*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n]*)/i,/^(?:\s*[^:\n\s\-]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:[\s\S]*?end note\b)/i,/^(?:stateDiagram\s+)/i,/^(?:stateDiagram-v2\s+)/i,/^(?:hide empty description\b)/i,/^(?:\[\*\])/i,/^(?:[^:\n\s\-\{]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:-->)/i,/^(?:--)/i,/^(?::::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{LINE:{rules:[9,10],inclusive:!1},struct:{rules:[9,10,22,26,29,35,42,43,44,45,54,55,56,57,71,72,73,74,75],inclusive:!1},FLOATING_NOTE_ID:{rules:[64],inclusive:!1},FLOATING_NOTE:{rules:[61,62,63],inclusive:!1},NOTE_TEXT:{rules:[66,67],inclusive:!1},NOTE_ID:{rules:[65],inclusive:!1},NOTE:{rules:[58,59,60],inclusive:!1},STYLEDEF_STYLEOPTS:{rules:[],inclusive:!1},STYLEDEF_STYLES:{rules:[31],inclusive:!1},STYLE_IDS:{rules:[],inclusive:!1},STYLE:{rules:[30],inclusive:!1},CLASS_STYLE:{rules:[28],inclusive:!1},CLASS:{rules:[27],inclusive:!1},CLASSDEFID:{rules:[25],inclusive:!1},CLASSDEF:{rules:[23,24],inclusive:!1},acc_descr_multiline:{rules:[20,21],inclusive:!1},acc_descr:{rules:[18],inclusive:!1},acc_title:{rules:[16],inclusive:!1},SCALE:{rules:[13,14,33,34],inclusive:!1},ALIAS:{rules:[],inclusive:!1},STATE_ID:{rules:[48],inclusive:!1},STATE_STRING:{rules:[49,50],inclusive:!1},FORK_STATE:{rules:[],inclusive:!1},STATE:{rules:[9,10,36,37,38,39,40,41,46,47,51,52,53],inclusive:!1},ID:{rules:[9,10],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,8,10,11,12,15,17,19,22,26,29,32,35,53,57,68,69,70,71,72,73,74,76,77,78],inclusive:!0}}}}();function nt(){this.yy={}}return pt.lexer=Wt,l(nt,"Parser"),nt.prototype=pt,pt.Parser=nt,new nt}();mt.parser=mt;var he=mt,ct="state",St="relation",J="default",vt="divider",It="fill:none",Lt="fill: #333",At="text",wt="normal",ft="rect",_t="rectWithTitle",Ot="divider",Nt="roundedWithTitle",K="statediagram",de=`${K}-state`,Rt="transition",ue=`${Rt} note-edge`,pe=`${K}-note`,ye=`${K}-cluster`,ge=`${K}-cluster-alt`,Bt="parent",Ft="note",bt="----",me=`${bt}${Ft}`,Pt=`${bt}${Bt}`,Yt=l((e,t="TB")=>{if(!e.doc)return t;let s=t;for(const a of e.doc)a.stmt==="dir"&&(s=a.value);return s},"getDir"),Se={getClasses:l(function(e,t){return t.db.getClasses()},"getClasses"),draw:l(async function(e,t,s,a){E.info("REF0:"),E.info("Drawing state diagram (v2)",t);const{securityLevel:o,state:i,layout:g}=$();a.db.extract(a.db.getRootDocV2());const d=a.db.getData(),c=Qt(t,o);d.type=a.type,d.layoutAlgorithm=g,d.nodeSpacing=(i==null?void 0:i.nodeSpacing)||50,d.rankSpacing=(i==null?void 0:i.rankSpacing)||50,d.markers=["barb"],d.diagramId=t,await te(d,c),ee.insertTitle(c,"statediagramTitleText",(i==null?void 0:i.titleTopMargin)??25,a.db.getDiagramTitle()),Zt(c,8,K,(i==null?void 0:i.useMaxWidth)??!0)},"draw"),getDir:Yt},lt=new Map,B=0;function ht(e="",t=0,s="",a=bt){return`state-${e}${s!==null&&s.length>0?`${a}${s}`:""}-${t}`}l(ht,"stateDomId");var fe=l((e,t,s,a,o,i,g,d)=>{E.trace("items",t),t.forEach(c=>{switch(c.stmt){case ct:case J:dt(e,c,s,a,o,i,g,d);break;case St:{dt(e,c.state1,s,a,o,i,g,d),dt(e,c.state2,s,a,o,i,g,d);const S={id:"edge"+B,start:c.state1.id,end:c.state2.id,arrowhead:"normal",arrowTypeEnd:"arrow_barb",style:It,labelStyle:"",label:U.sanitizeText(c.description,$()),arrowheadStyle:Lt,labelpos:"c",labelType:At,thickness:wt,classes:Rt,look:g};o.push(S),B++}}})},"setupDoc"),Gt=l((e,t="TB")=>{let s=t;if(e.doc)for(const a of e.doc)a.stmt==="dir"&&(s=a.value);return s},"getDir");function q(e,t,s){if(!t.id||t.id==="</join></fork>"||t.id==="</choice>")return;t.cssClasses&&(Array.isArray(t.cssCompiledStyles)||(t.cssCompiledStyles=[]),t.cssClasses.split(" ").forEach(o=>{if(s.get(o)){const i=s.get(o);t.cssCompiledStyles=[...t.cssCompiledStyles,...i.styles]}}));const a=e.find(o=>o.id===t.id);a?Object.assign(a,t):e.push(t)}function jt(e){var t;return((t=e==null?void 0:e.classes)==null?void 0:t.join(" "))??""}function zt(e){return(e==null?void 0:e.styles)??[]}l(q,"insertOrUpdateNode"),l(jt,"getClassesFromDbInfo"),l(zt,"getStylesFromDbInfo");var dt=l((e,t,s,a,o,i,g,d)=>{var F,P;const c=t.id,S=s.get(c),b=jt(S),v=zt(S);if(E.info("dataFetcher parsedItem",t,S,v),c!=="root"){let L=ft;t.start===!0?L="stateStart":t.start===!1&&(L="stateEnd"),t.type!==J&&(L=t.type),lt.get(c)||lt.set(c,{id:c,shape:L,description:U.sanitizeText(c,$()),cssClasses:`${b} ${de}`,cssStyles:v});const p=lt.get(c);t.description&&(Array.isArray(p.description)?(p.shape=_t,p.description.push(t.description)):((F=p.description)==null?void 0:F.length)>0?(p.shape=_t,p.description===c?p.description=[t.description]:p.description=[p.description,t.description]):(p.shape=ft,p.description=t.description),p.description=U.sanitizeTextOrArray(p.description,$())),((P=p.description)==null?void 0:P.length)===1&&p.shape===_t&&(p.type==="group"?p.shape=Nt:p.shape=ft),!p.type&&t.doc&&(E.info("Setting cluster for XCX",c,Gt(t)),p.type="group",p.isGroup=!0,p.dir=Gt(t),p.shape=t.type===vt?Ot:Nt,p.cssClasses=`${p.cssClasses} ${ye} ${i?ge:""}`);const I={labelStyle:"",shape:p.shape,label:p.description,cssClasses:p.cssClasses,cssCompiledStyles:[],cssStyles:p.cssStyles,id:c,dir:p.dir,domId:ht(c,B),type:p.type,isGroup:p.type==="group",padding:8,rx:10,ry:10,look:g};if(I.shape===Ot&&(I.label=""),e&&e.id!=="root"&&(E.trace("Setting node ",c," to be child of its parent ",e.id),I.parentId=e.id),I.centerLabel=!0,t.note){const A={labelStyle:"",shape:"note",label:t.note.text,cssClasses:pe,cssStyles:[],cssCompilesStyles:[],id:c+me+"-"+B,domId:ht(c,B,Ft),type:p.type,isGroup:p.type==="group",padding:$().flowchart.padding,look:g,position:t.note.position},Y=c+Pt,G={labelStyle:"",shape:"noteGroup",label:t.note.text,cssClasses:p.cssClasses,cssStyles:[],id:c+Pt,domId:ht(c,B,Bt),type:"group",isGroup:!0,padding:16,look:g,position:t.note.position};B++,G.id=Y,A.parentId=Y,q(a,G,d),q(a,A,d),q(a,I,d);let N=c,R=A.id;t.note.position==="left of"&&(N=A.id,R=c),o.push({id:N+"-"+R,start:N,end:R,arrowhead:"none",arrowTypeEnd:"",style:It,labelStyle:"",classes:ue,arrowheadStyle:Lt,labelpos:"c",labelType:At,thickness:wt,look:g})}else q(a,I,d)}t.doc&&(E.trace("Adding nodes children "),fe(t,t.doc,s,a,o,!i,g,d))},"dataFetcher"),_e=l(()=>{lt.clear(),B=0},"reset"),Tt="[*]",Ut="start",Mt=Tt,Xt="color",Ht="fill";function kt(){return new Map}l(kt,"newClassesList");var Vt=l(()=>({relations:[],states:new Map,documents:{}}),"newDoc"),ut=l(e=>JSON.parse(JSON.stringify(e)),"clone"),be=(M=class{constructor(t){k(this,"version");k(this,"nodes",[]);k(this,"edges",[]);k(this,"rootDoc",[]);k(this,"classes",kt());k(this,"documents",{root:Vt()});k(this,"currentDocument",this.documents.root);k(this,"startEndCount",0);k(this,"dividerCnt",0);k(this,"getAccTitle",ne);k(this,"setAccTitle",re);k(this,"getAccDescription",ae);k(this,"setAccDescription",oe);k(this,"setDiagramTitle",ce);k(this,"getDiagramTitle",le);this.clear(),this.version=t,this.setRootDoc=this.setRootDoc.bind(this),this.getDividerId=this.getDividerId.bind(this),this.setDirection=this.setDirection.bind(this),this.trimColon=this.trimColon.bind(this)}setRootDoc(t){E.info("Setting root doc",t),this.rootDoc=t,this.version===1?this.extract(t):this.extract(this.getRootDocV2())}getRootDoc(){return this.rootDoc}docTranslator(t,s,a){if(s.stmt===St)this.docTranslator(t,s.state1,!0),this.docTranslator(t,s.state2,!1);else if(s.stmt===ct&&(s.id==="[*]"?(s.id=a?t.id+"_start":t.id+"_end",s.start=a):s.id=s.id.trim()),s.doc){const o=[];let i,g=[];for(i=0;i<s.doc.length;i++)if(s.doc[i].type===vt){const d=ut(s.doc[i]);d.doc=ut(g),o.push(d),g=[]}else g.push(s.doc[i]);if(o.length>0&&g.length>0){const d={stmt:ct,id:se(),type:"divider",doc:ut(g)};o.push(ut(d)),s.doc=o}s.doc.forEach(d=>this.docTranslator(s,d,!0))}}getRootDocV2(){return this.docTranslator({id:"root"},{id:"root",doc:this.rootDoc},!0),{id:"root",doc:this.rootDoc}}extract(t){let s;s=t.doc?t.doc:t,E.info(s),this.clear(!0),E.info("Extract initial document:",s),s.forEach(i=>{switch(E.warn("Statement",i.stmt),i.stmt){case ct:this.addState(i.id.trim(),i.type,i.doc,i.description,i.note,i.classes,i.styles,i.textStyles);break;case St:this.addRelation(i.state1,i.state2,i.description);break;case"classDef":this.addStyleClass(i.id.trim(),i.classes);break;case"style":{const g=i.id.trim().split(","),d=i.styleClass.split(",");g.forEach(c=>{let S=this.getState(c);if(S===void 0){const b=c.trim();this.addState(b),S=this.getState(b)}S.styles=d.map(b=>{var v;return(v=b.replace(/;/g,""))==null?void 0:v.trim()})})}break;case"applyClass":this.setCssClass(i.id.trim(),i.styleClass)}});const a=this.getStates(),o=$().look;_e(),dt(void 0,this.getRootDocV2(),a,this.nodes,this.edges,!0,o,this.classes),this.nodes.forEach(i=>{if(Array.isArray(i.label)){if(i.description=i.label.slice(1),i.isGroup&&i.description.length>0)throw new Error("Group nodes can only have label. Remove the additional description for node ["+i.id+"]");i.label=i.label[0]}})}addState(t,s=J,a=null,o=null,i=null,g=null,d=null,c=null){const S=t==null?void 0:t.trim();if(this.currentDocument.states.has(S)?(this.currentDocument.states.get(S).doc||(this.currentDocument.states.get(S).doc=a),this.currentDocument.states.get(S).type||(this.currentDocument.states.get(S).type=s)):(E.info("Adding state ",S,o),this.currentDocument.states.set(S,{id:S,descriptions:[],type:s,doc:a,note:i,classes:[],styles:[],textStyles:[]})),o&&(E.info("Setting state description",S,o),typeof o=="string"&&this.addDescription(S,o.trim()),typeof o=="object"&&o.forEach(b=>this.addDescription(S,b.trim()))),i){const b=this.currentDocument.states.get(S);b.note=i,b.note.text=U.sanitizeText(b.note.text,$())}g&&(E.info("Setting state classes",S,g),(typeof g=="string"?[g]:g).forEach(b=>this.setCssClass(S,b.trim()))),d&&(E.info("Setting state styles",S,d),(typeof d=="string"?[d]:d).forEach(b=>this.setStyle(S,b.trim()))),c&&(E.info("Setting state styles",S,d),(typeof c=="string"?[c]:c).forEach(b=>this.setTextStyle(S,b.trim())))}clear(t){this.nodes=[],this.edges=[],this.documents={root:Vt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.classes=kt(),t||ie()}getState(t){return this.currentDocument.states.get(t)}getStates(){return this.currentDocument.states}logDocuments(){E.info("Documents = ",this.documents)}getRelations(){return this.currentDocument.relations}startIdIfNeeded(t=""){let s=t;return t===Tt&&(this.startEndCount++,s=`${Ut}${this.startEndCount}`),s}startTypeIfNeeded(t="",s=J){return t===Tt?Ut:s}endIdIfNeeded(t=""){let s=t;return t===Mt&&(this.startEndCount++,s=`end${this.startEndCount}`),s}endTypeIfNeeded(t="",s=J){return t===Mt?"end":s}addRelationObjs(t,s,a){let o=this.startIdIfNeeded(t.id.trim()),i=this.startTypeIfNeeded(t.id.trim(),t.type),g=this.startIdIfNeeded(s.id.trim()),d=this.startTypeIfNeeded(s.id.trim(),s.type);this.addState(o,i,t.doc,t.description,t.note,t.classes,t.styles,t.textStyles),this.addState(g,d,s.doc,s.description,s.note,s.classes,s.styles,s.textStyles),this.currentDocument.relations.push({id1:o,id2:g,relationTitle:U.sanitizeText(a,$())})}addRelation(t,s,a){if(typeof t=="object")this.addRelationObjs(t,s,a);else{const o=this.startIdIfNeeded(t.trim()),i=this.startTypeIfNeeded(t),g=this.endIdIfNeeded(s.trim()),d=this.endTypeIfNeeded(s);this.addState(o,i),this.addState(g,d),this.currentDocument.relations.push({id1:o,id2:g,title:U.sanitizeText(a,$())})}}addDescription(t,s){const a=this.currentDocument.states.get(t),o=s.startsWith(":")?s.replace(":","").trim():s;a.descriptions.push(U.sanitizeText(o,$()))}cleanupLabel(t){return t.substring(0,1)===":"?t.substr(2).trim():t.trim()}getDividerId(){return this.dividerCnt++,"divider-id-"+this.dividerCnt}addStyleClass(t,s=""){this.classes.has(t)||this.classes.set(t,{id:t,styles:[],textStyles:[]});const a=this.classes.get(t);s!=null&&s.split(",").forEach(o=>{const i=o.replace(/([^;]*);/,"$1").trim();if(RegExp(Xt).exec(o)){const g=i.replace(Ht,"bgFill").replace(Xt,Ht);a.textStyles.push(g)}a.styles.push(i)})}getClasses(){return this.classes}setCssClass(t,s){t.split(",").forEach(a=>{let o=this.getState(a);if(o===void 0){const i=a.trim();this.addState(i),o=this.getState(i)}o.classes.push(s)})}setStyle(t,s){const a=this.getState(t);a!==void 0&&a.styles.push(s)}setTextStyle(t,s){const a=this.getState(t);a!==void 0&&a.textStyles.push(s)}getDirectionStatement(){return this.rootDoc.find(t=>t.stmt==="dir")}getDirection(){var t;return((t=this.getDirectionStatement())==null?void 0:t.value)??"TB"}setDirection(t){const s=this.getDirectionStatement();s?s.value=t:this.rootDoc.unshift({stmt:"dir",value:t})}trimColon(t){return t&&t[0]===":"?t.substr(1).trim():t.trim()}getData(){const t=$();return{nodes:this.nodes,edges:this.edges,other:{},config:t,direction:Yt(this.getRootDocV2())}}getConfig(){return $().state}},l(M,"StateDB"),k(M,"relationType",{AGGREGATION:0,EXTENSION:1,COMPOSITION:2,DEPENDENCY:3}),M),Te=l(e=>`
defs #statediagram-barbEnd {
    fill: ${e.transitionColor};
    stroke: ${e.transitionColor};
  }
g.stateGroup text {
  fill: ${e.nodeBorder};
  stroke: none;
  font-size: 10px;
}
g.stateGroup text {
  fill: ${e.textColor};
  stroke: none;
  font-size: 10px;

}
g.stateGroup .state-title {
  font-weight: bolder;
  fill: ${e.stateLabelColor};
}

g.stateGroup rect {
  fill: ${e.mainBkg};
  stroke: ${e.nodeBorder};
}

g.stateGroup line {
  stroke: ${e.lineColor};
  stroke-width: 1;
}

.transition {
  stroke: ${e.transitionColor};
  stroke-width: 1;
  fill: none;
}

.stateGroup .composit {
  fill: ${e.background};
  border-bottom: 1px
}

.stateGroup .alt-composit {
  fill: #e0e0e0;
  border-bottom: 1px
}

.state-note {
  stroke: ${e.noteBorderColor};
  fill: ${e.noteBkgColor};

  text {
    fill: ${e.noteTextColor};
    stroke: none;
    font-size: 10px;
  }
}

.stateLabel .box {
  stroke: none;
  stroke-width: 0;
  fill: ${e.mainBkg};
  opacity: 0.5;
}

.edgeLabel .label rect {
  fill: ${e.labelBackgroundColor};
  opacity: 0.5;
}
.edgeLabel {
  background-color: ${e.edgeLabelBackground};
  p {
    background-color: ${e.edgeLabelBackground};
  }
  rect {
    opacity: 0.5;
    background-color: ${e.edgeLabelBackground};
    fill: ${e.edgeLabelBackground};
  }
  text-align: center;
}
.edgeLabel .label text {
  fill: ${e.transitionLabelColor||e.tertiaryTextColor};
}
.label div .edgeLabel {
  color: ${e.transitionLabelColor||e.tertiaryTextColor};
}

.stateLabel text {
  fill: ${e.stateLabelColor};
  font-size: 10px;
  font-weight: bold;
}

.node circle.state-start {
  fill: ${e.specialStateColor};
  stroke: ${e.specialStateColor};
}

.node .fork-join {
  fill: ${e.specialStateColor};
  stroke: ${e.specialStateColor};
}

.node circle.state-end {
  fill: ${e.innerEndBackground};
  stroke: ${e.background};
  stroke-width: 1.5
}
.end-state-inner {
  fill: ${e.compositeBackground||e.background};
  // stroke: ${e.background};
  stroke-width: 1.5
}

.node rect {
  fill: ${e.stateBkg||e.mainBkg};
  stroke: ${e.stateBorder||e.nodeBorder};
  stroke-width: 1px;
}
.node polygon {
  fill: ${e.mainBkg};
  stroke: ${e.stateBorder||e.nodeBorder};;
  stroke-width: 1px;
}
#statediagram-barbEnd {
  fill: ${e.lineColor};
}

.statediagram-cluster rect {
  fill: ${e.compositeTitleBackground};
  stroke: ${e.stateBorder||e.nodeBorder};
  stroke-width: 1px;
}

.cluster-label, .nodeLabel {
  color: ${e.stateLabelColor};
  // line-height: 1;
}

.statediagram-cluster rect.outer {
  rx: 5px;
  ry: 5px;
}
.statediagram-state .divider {
  stroke: ${e.stateBorder||e.nodeBorder};
}

.statediagram-state .title-state {
  rx: 5px;
  ry: 5px;
}
.statediagram-cluster.statediagram-cluster .inner {
  fill: ${e.compositeBackground||e.background};
}
.statediagram-cluster.statediagram-cluster-alt .inner {
  fill: ${e.altBackground?e.altBackground:"#efefef"};
}

.statediagram-cluster .inner {
  rx:0;
  ry:0;
}

.statediagram-state rect.basic {
  rx: 5px;
  ry: 5px;
}
.statediagram-state rect.divider {
  stroke-dasharray: 10,10;
  fill: ${e.altBackground?e.altBackground:"#efefef"};
}

.note-edge {
  stroke-dasharray: 5;
}

.statediagram-note rect {
  fill: ${e.noteBkgColor};
  stroke: ${e.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}
.statediagram-note rect {
  fill: ${e.noteBkgColor};
  stroke: ${e.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}

.statediagram-note text {
  fill: ${e.noteTextColor};
}

.statediagram-note .nodeLabel {
  color: ${e.noteTextColor};
}
.statediagram .edgeLabel {
  color: red; // ${e.noteTextColor};
}

#dependencyStart, #dependencyEnd {
  fill: ${e.lineColor};
  stroke: ${e.lineColor};
  stroke-width: 1;
}

.statediagramTitleText {
  text-anchor: middle;
  font-size: 18px;
  fill: ${e.textColor};
}
`,"getStyles");export{be as S,he as a,Se as b,Te as s};
