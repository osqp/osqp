var W;import{g as Qt}from"./chunk-WVR4S24B-DkjWv9uk.js";import{s as Zt}from"./chunk-NRVI72HA-gO-7Uy2L.js";import{_ as d,l as k,c as R,r as te,u as ee,a as se,b as ie,g as ne,s as re,p as ae,q as oe,S as ce,j as U,y as le}from"./mermaid-D59lkToe.js";var mt=function(){var e=d(function(a,y,p,T){for(p=p||{},T=a.length;T--;p[a[T]]=y);return p},"o"),t=[1,2],s=[1,3],r=[1,4],i=[2,4],o=[1,9],l=[1,11],g=[1,16],u=[1,17],m=[1,18],S=[1,19],C=[1,33],v=[1,20],D=[1,21],w=[1,22],x=[1,23],L=[1,24],h=[1,26],E=[1,27],N=[1,28],O=[1,29],j=[1,30],P=[1,31],Y=[1,32],et=[1,35],st=[1,36],it=[1,37],nt=[1,38],H=[1,34],f=[1,4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],rt=[1,4,5,14,15,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,39,40,41,45,48,51,52,53,54,57],Et=[4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],yt={trace:d(function(){},"trace"),yy:{},symbols_:{error:2,start:3,SPACE:4,NL:5,SD:6,document:7,line:8,statement:9,classDefStatement:10,styleStatement:11,cssClassStatement:12,idStatement:13,DESCR:14,"-->":15,HIDE_EMPTY:16,scale:17,WIDTH:18,COMPOSIT_STATE:19,STRUCT_START:20,STRUCT_STOP:21,STATE_DESCR:22,AS:23,ID:24,FORK:25,JOIN:26,CHOICE:27,CONCURRENT:28,note:29,notePosition:30,NOTE_TEXT:31,direction:32,acc_title:33,acc_title_value:34,acc_descr:35,acc_descr_value:36,acc_descr_multiline_value:37,CLICK:38,STRING:39,HREF:40,classDef:41,CLASSDEF_ID:42,CLASSDEF_STYLEOPTS:43,DEFAULT:44,style:45,STYLE_IDS:46,STYLEDEF_STYLEOPTS:47,class:48,CLASSENTITY_IDS:49,STYLECLASS:50,direction_tb:51,direction_bt:52,direction_rl:53,direction_lr:54,eol:55,";":56,EDGE_STATE:57,STYLE_SEPARATOR:58,left_of:59,right_of:60,$accept:0,$end:1},terminals_:{2:"error",4:"SPACE",5:"NL",6:"SD",14:"DESCR",15:"-->",16:"HIDE_EMPTY",17:"scale",18:"WIDTH",19:"COMPOSIT_STATE",20:"STRUCT_START",21:"STRUCT_STOP",22:"STATE_DESCR",23:"AS",24:"ID",25:"FORK",26:"JOIN",27:"CHOICE",28:"CONCURRENT",29:"note",31:"NOTE_TEXT",33:"acc_title",34:"acc_title_value",35:"acc_descr",36:"acc_descr_value",37:"acc_descr_multiline_value",38:"CLICK",39:"STRING",40:"HREF",41:"classDef",42:"CLASSDEF_ID",43:"CLASSDEF_STYLEOPTS",44:"DEFAULT",45:"style",46:"STYLE_IDS",47:"STYLEDEF_STYLEOPTS",48:"class",49:"CLASSENTITY_IDS",50:"STYLECLASS",51:"direction_tb",52:"direction_bt",53:"direction_rl",54:"direction_lr",56:";",57:"EDGE_STATE",58:"STYLE_SEPARATOR",59:"left_of",60:"right_of"},productions_:[0,[3,2],[3,2],[3,2],[7,0],[7,2],[8,2],[8,1],[8,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,3],[9,4],[9,1],[9,2],[9,1],[9,4],[9,3],[9,6],[9,1],[9,1],[9,1],[9,1],[9,4],[9,4],[9,1],[9,2],[9,2],[9,1],[9,5],[9,5],[10,3],[10,3],[11,3],[12,3],[32,1],[32,1],[32,1],[32,1],[55,1],[55,1],[13,1],[13,1],[13,3],[13,3],[30,1],[30,1]],performAction:d(function(a,y,p,T,_,n,q){var c=n.length-1;switch(_){case 3:return T.setRootDoc(n[c]),n[c];case 4:this.$=[];break;case 5:n[c]!="nl"&&(n[c-1].push(n[c]),this.$=n[c-1]);break;case 6:case 7:case 12:this.$=n[c];break;case 8:this.$="nl";break;case 13:const b=n[c-1];b.description=T.trimColon(n[c]),this.$=b;break;case 14:this.$={stmt:"relation",state1:n[c-2],state2:n[c]};break;case 15:const B=T.trimColon(n[c]);this.$={stmt:"relation",state1:n[c-3],state2:n[c-1],description:B};break;case 19:this.$={stmt:"state",id:n[c-3],type:"default",description:"",doc:n[c-1]};break;case 20:var z=n[c],X=n[c-2].trim();if(n[c].match(":")){var ot=n[c].split(":");z=ot[0],X=[X,ot[1]]}this.$={stmt:"state",id:z,type:"default",description:X};break;case 21:this.$={stmt:"state",id:n[c-3],type:"default",description:n[c-5],doc:n[c-1]};break;case 22:this.$={stmt:"state",id:n[c],type:"fork"};break;case 23:this.$={stmt:"state",id:n[c],type:"join"};break;case 24:this.$={stmt:"state",id:n[c],type:"choice"};break;case 25:this.$={stmt:"state",id:T.getDividerId(),type:"divider"};break;case 26:this.$={stmt:"state",id:n[c-1].trim(),note:{position:n[c-2].trim(),text:n[c].trim()}};break;case 29:this.$=n[c].trim(),T.setAccTitle(this.$);break;case 30:case 31:this.$=n[c].trim(),T.setAccDescription(this.$);break;case 32:this.$={stmt:"click",id:n[c-3],url:n[c-2],tooltip:n[c-1]};break;case 33:this.$={stmt:"click",id:n[c-3],url:n[c-1],tooltip:""};break;case 34:case 35:this.$={stmt:"classDef",id:n[c-1].trim(),classes:n[c].trim()};break;case 36:this.$={stmt:"style",id:n[c-1].trim(),styleClass:n[c].trim()};break;case 37:this.$={stmt:"applyClass",id:n[c-1].trim(),styleClass:n[c].trim()};break;case 38:T.setDirection("TB"),this.$={stmt:"dir",value:"TB"};break;case 39:T.setDirection("BT"),this.$={stmt:"dir",value:"BT"};break;case 40:T.setDirection("RL"),this.$={stmt:"dir",value:"RL"};break;case 41:T.setDirection("LR"),this.$={stmt:"dir",value:"LR"};break;case 44:case 45:this.$={stmt:"state",id:n[c].trim(),type:"default",description:""};break;case 46:case 47:this.$={stmt:"state",id:n[c-2].trim(),classes:[n[c].trim()],type:"default",description:""}}},"anonymous"),table:[{3:1,4:t,5:s,6:r},{1:[3]},{3:5,4:t,5:s,6:r},{3:6,4:t,5:s,6:r},e([1,4,5,16,17,19,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],i,{7:7}),{1:[2,1]},{1:[2,2]},{1:[2,3],4:o,5:l,8:8,9:10,10:12,11:13,12:14,13:15,16:g,17:u,19:m,22:S,24:C,25:v,26:D,27:w,28:x,29:L,32:25,33:h,35:E,37:N,38:O,41:j,45:P,48:Y,51:et,52:st,53:it,54:nt,57:H},e(f,[2,5]),{9:39,10:12,11:13,12:14,13:15,16:g,17:u,19:m,22:S,24:C,25:v,26:D,27:w,28:x,29:L,32:25,33:h,35:E,37:N,38:O,41:j,45:P,48:Y,51:et,52:st,53:it,54:nt,57:H},e(f,[2,7]),e(f,[2,8]),e(f,[2,9]),e(f,[2,10]),e(f,[2,11]),e(f,[2,12],{14:[1,40],15:[1,41]}),e(f,[2,16]),{18:[1,42]},e(f,[2,18],{20:[1,43]}),{23:[1,44]},e(f,[2,22]),e(f,[2,23]),e(f,[2,24]),e(f,[2,25]),{30:45,31:[1,46],59:[1,47],60:[1,48]},e(f,[2,28]),{34:[1,49]},{36:[1,50]},e(f,[2,31]),{13:51,24:C,57:H},{42:[1,52],44:[1,53]},{46:[1,54]},{49:[1,55]},e(rt,[2,44],{58:[1,56]}),e(rt,[2,45],{58:[1,57]}),e(f,[2,38]),e(f,[2,39]),e(f,[2,40]),e(f,[2,41]),e(f,[2,6]),e(f,[2,13]),{13:58,24:C,57:H},e(f,[2,17]),e(Et,i,{7:59}),{24:[1,60]},{24:[1,61]},{23:[1,62]},{24:[2,48]},{24:[2,49]},e(f,[2,29]),e(f,[2,30]),{39:[1,63],40:[1,64]},{43:[1,65]},{43:[1,66]},{47:[1,67]},{50:[1,68]},{24:[1,69]},{24:[1,70]},e(f,[2,14],{14:[1,71]}),{4:o,5:l,8:8,9:10,10:12,11:13,12:14,13:15,16:g,17:u,19:m,21:[1,72],22:S,24:C,25:v,26:D,27:w,28:x,29:L,32:25,33:h,35:E,37:N,38:O,41:j,45:P,48:Y,51:et,52:st,53:it,54:nt,57:H},e(f,[2,20],{20:[1,73]}),{31:[1,74]},{24:[1,75]},{39:[1,76]},{39:[1,77]},e(f,[2,34]),e(f,[2,35]),e(f,[2,36]),e(f,[2,37]),e(rt,[2,46]),e(rt,[2,47]),e(f,[2,15]),e(f,[2,19]),e(Et,i,{7:78}),e(f,[2,26]),e(f,[2,27]),{5:[1,79]},{5:[1,80]},{4:o,5:l,8:8,9:10,10:12,11:13,12:14,13:15,16:g,17:u,19:m,21:[1,81],22:S,24:C,25:v,26:D,27:w,28:x,29:L,32:25,33:h,35:E,37:N,38:O,41:j,45:P,48:Y,51:et,52:st,53:it,54:nt,57:H},e(f,[2,32]),e(f,[2,33]),e(f,[2,21])],defaultActions:{5:[2,1],6:[2,2],47:[2,48],48:[2,49]},parseError:d(function(a,y){if(!y.recoverable){var p=new Error(a);throw p.hash=y,p}this.trace(a)},"parseError"),parse:d(function(a){var y=this,p=[0],T=[],_=[null],n=[],q=this.table,c="",z=0,X=0,ot=n.slice.call(arguments,1),b=Object.create(this.lexer),B={yy:{}};for(var gt in this.yy)Object.prototype.hasOwnProperty.call(this.yy,gt)&&(B.yy[gt]=this.yy[gt]);b.setInput(a,B.yy),B.yy.lexer=b,B.yy.parser=this,b.yylloc===void 0&&(b.yylloc={});var ft=b.yylloc;n.push(ft);var qt=b.options&&b.options.ranges;function Ct(){var A;return typeof(A=T.pop()||b.lex()||1)!="number"&&(A instanceof Array&&(A=(T=A).pop()),A=y.symbols_[A]||A),A}typeof B.yy.parseError=="function"?this.parseError=B.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError,d(function(A){p.length=p.length-2*A,_.length=_.length-A,n.length=n.length-A},"popStack"),d(Ct,"lex");for(var $,M,I,Dt,ct,F,xt,lt,V={};;){if(M=p[p.length-1],this.defaultActions[M]?I=this.defaultActions[M]:($==null&&($=Ct()),I=q[M]&&q[M][$]),I===void 0||!I.length||!I[0]){var $t="";for(ct in lt=[],q[M])this.terminals_[ct]&&ct>2&&lt.push("'"+this.terminals_[ct]+"'");$t=b.showPosition?"Parse error on line "+(z+1)+`:
`+b.showPosition()+`
Expecting `+lt.join(", ")+", got '"+(this.terminals_[$]||$)+"'":"Parse error on line "+(z+1)+": Unexpected "+($==1?"end of input":"'"+(this.terminals_[$]||$)+"'"),this.parseError($t,{text:b.match,token:this.terminals_[$]||$,line:b.yylineno,loc:ft,expected:lt})}if(I[0]instanceof Array&&I.length>1)throw new Error("Parse Error: multiple actions possible at state: "+M+", token: "+$);switch(I[0]){case 1:p.push($),_.push(b.yytext),n.push(b.yylloc),p.push(I[1]),$=null,X=b.yyleng,c=b.yytext,z=b.yylineno,ft=b.yylloc;break;case 2:if(F=this.productions_[I[1]][1],V.$=_[_.length-F],V._$={first_line:n[n.length-(F||1)].first_line,last_line:n[n.length-1].last_line,first_column:n[n.length-(F||1)].first_column,last_column:n[n.length-1].last_column},qt&&(V._$.range=[n[n.length-(F||1)].range[0],n[n.length-1].range[1]]),(Dt=this.performAction.apply(V,[c,X,z,B.yy,I[1],_,n].concat(ot)))!==void 0)return Dt;F&&(p=p.slice(0,-1*F*2),_=_.slice(0,-1*F),n=n.slice(0,-1*F)),p.push(this.productions_[I[1]][0]),_.push(V.$),n.push(V._$),xt=q[p[p.length-2]][p[p.length-1]],p.push(xt);break;case 3:return!0}}return!0},"parse")},Kt=function(){return{EOF:1,parseError:d(function(a,y){if(!this.yy.parser)throw new Error(a);this.yy.parser.parseError(a,y)},"parseError"),setInput:d(function(a,y){return this.yy=y||this.yy||{},this._input=a,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:d(function(){var a=this._input[0];return this.yytext+=a,this.yyleng++,this.offset++,this.match+=a,this.matched+=a,a.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),a},"input"),unput:d(function(a){var y=a.length,p=a.split(/(?:\r\n?|\n)/g);this._input=a+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-y),this.offset-=y;var T=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),p.length-1&&(this.yylineno-=p.length-1);var _=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:p?(p.length===T.length?this.yylloc.first_column:0)+T[T.length-p.length].length-p[0].length:this.yylloc.first_column-y},this.options.ranges&&(this.yylloc.range=[_[0],_[0]+this.yyleng-y]),this.yyleng=this.yytext.length,this},"unput"),more:d(function(){return this._more=!0,this},"more"),reject:d(function(){return this.options.backtrack_lexer?(this._backtrack=!0,this):this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"reject"),less:d(function(a){this.unput(this.match.slice(a))},"less"),pastInput:d(function(){var a=this.matched.substr(0,this.matched.length-this.match.length);return(a.length>20?"...":"")+a.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:d(function(){var a=this.match;return a.length<20&&(a+=this._input.substr(0,20-a.length)),(a.substr(0,20)+(a.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:d(function(){var a=this.pastInput(),y=new Array(a.length+1).join("-");return a+this.upcomingInput()+`
`+y+"^"},"showPosition"),test_match:d(function(a,y){var p,T,_;if(this.options.backtrack_lexer&&(_={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(_.yylloc.range=this.yylloc.range.slice(0))),(T=a[0].match(/(?:\r\n?|\n).*/g))&&(this.yylineno+=T.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:T?T[T.length-1].length-T[T.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+a[0].length},this.yytext+=a[0],this.match+=a[0],this.matches=a,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(a[0].length),this.matched+=a[0],p=this.performAction.call(this,this.yy,this,y,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),p)return p;if(this._backtrack){for(var n in _)this[n]=_[n];return!1}return!1},"test_match"),next:d(function(){if(this.done)return this.EOF;var a,y,p,T;this._input||(this.done=!0),this._more||(this.yytext="",this.match="");for(var _=this._currentRules(),n=0;n<_.length;n++)if((p=this._input.match(this.rules[_[n]]))&&(!y||p[0].length>y[0].length)){if(y=p,T=n,this.options.backtrack_lexer){if((a=this.test_match(p,_[n]))!==!1)return a;if(this._backtrack){y=!1;continue}return!1}if(!this.options.flex)break}return y?(a=this.test_match(y,_[T]))!==!1&&a:this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:d(function(){var a=this.next();return a||this.lex()},"lex"),begin:d(function(a){this.conditionStack.push(a)},"begin"),popState:d(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:d(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:d(function(a){return(a=this.conditionStack.length-1-Math.abs(a||0))>=0?this.conditionStack[a]:"INITIAL"},"topState"),pushState:d(function(a){this.begin(a)},"pushState"),stateStackSize:d(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:d(function(a,y,p,T){switch(p){case 0:return 38;case 1:return 40;case 2:return 39;case 3:return 44;case 4:case 45:return 51;case 5:case 46:return 52;case 6:case 47:return 53;case 7:case 48:return 54;case 8:case 9:case 11:case 12:case 13:case 14:case 57:case 59:case 65:break;case 10:case 80:return 5;case 15:case 35:return this.pushState("SCALE"),17;case 16:case 36:return 18;case 17:case 23:case 37:case 52:case 55:this.popState();break;case 18:return this.begin("acc_title"),33;case 19:return this.popState(),"acc_title_value";case 20:return this.begin("acc_descr"),35;case 21:return this.popState(),"acc_descr_value";case 22:this.begin("acc_descr_multiline");break;case 24:return"acc_descr_multiline_value";case 25:return this.pushState("CLASSDEF"),41;case 26:return this.popState(),this.pushState("CLASSDEFID"),"DEFAULT_CLASSDEF_ID";case 27:return this.popState(),this.pushState("CLASSDEFID"),42;case 28:return this.popState(),43;case 29:return this.pushState("CLASS"),48;case 30:return this.popState(),this.pushState("CLASS_STYLE"),49;case 31:return this.popState(),50;case 32:return this.pushState("STYLE"),45;case 33:return this.popState(),this.pushState("STYLEDEF_STYLES"),46;case 34:return this.popState(),47;case 38:this.pushState("STATE");break;case 39:case 42:return this.popState(),y.yytext=y.yytext.slice(0,-8).trim(),25;case 40:case 43:return this.popState(),y.yytext=y.yytext.slice(0,-8).trim(),26;case 41:case 44:return this.popState(),y.yytext=y.yytext.slice(0,-10).trim(),27;case 49:this.pushState("STATE_STRING");break;case 50:return this.pushState("STATE_ID"),"AS";case 51:case 67:return this.popState(),"ID";case 53:return"STATE_DESCR";case 54:return 19;case 56:return this.popState(),this.pushState("struct"),20;case 58:return this.popState(),21;case 60:return this.begin("NOTE"),29;case 61:return this.popState(),this.pushState("NOTE_ID"),59;case 62:return this.popState(),this.pushState("NOTE_ID"),60;case 63:this.popState(),this.pushState("FLOATING_NOTE");break;case 64:return this.popState(),this.pushState("FLOATING_NOTE_ID"),"AS";case 66:return"NOTE_TEXT";case 68:return this.popState(),this.pushState("NOTE_TEXT"),24;case 69:return this.popState(),y.yytext=y.yytext.substr(2).trim(),31;case 70:return this.popState(),y.yytext=y.yytext.slice(0,-8).trim(),31;case 71:case 72:return 6;case 73:return 16;case 74:return 57;case 75:return 24;case 76:return y.yytext=y.yytext.trim(),14;case 77:return 15;case 78:return 28;case 79:return 58;case 81:return"INVALID"}},"anonymous"),rules:[/^(?:click\b)/i,/^(?:href\b)/i,/^(?:"[^"]*")/i,/^(?:default\b)/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:[\s]+)/i,/^(?:((?!\n)\s)+)/i,/^(?:#[^\n]*)/i,/^(?:%[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:classDef\s+)/i,/^(?:DEFAULT\s+)/i,/^(?:\w+\s+)/i,/^(?:[^\n]*)/i,/^(?:class\s+)/i,/^(?:(\w+)+((,\s*\w+)*))/i,/^(?:[^\n]*)/i,/^(?:style\s+)/i,/^(?:[\w,]+\s+)/i,/^(?:[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:state\s+)/i,/^(?:.*<<fork>>)/i,/^(?:.*<<join>>)/i,/^(?:.*<<choice>>)/i,/^(?:.*\[\[fork\]\])/i,/^(?:.*\[\[join\]\])/i,/^(?:.*\[\[choice\]\])/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:["])/i,/^(?:\s*as\s+)/i,/^(?:[^\n\{]*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n\s\{]+)/i,/^(?:\n)/i,/^(?:\{)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:\})/i,/^(?:[\n])/i,/^(?:note\s+)/i,/^(?:left of\b)/i,/^(?:right of\b)/i,/^(?:")/i,/^(?:\s*as\s*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n]*)/i,/^(?:\s*[^:\n\s\-]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:[\s\S]*?end note\b)/i,/^(?:stateDiagram\s+)/i,/^(?:stateDiagram-v2\s+)/i,/^(?:hide empty description\b)/i,/^(?:\[\*\])/i,/^(?:[^:\n\s\-\{]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:-->)/i,/^(?:--)/i,/^(?::::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{LINE:{rules:[12,13],inclusive:!1},struct:{rules:[12,13,25,29,32,38,45,46,47,48,57,58,59,60,74,75,76,77,78],inclusive:!1},FLOATING_NOTE_ID:{rules:[67],inclusive:!1},FLOATING_NOTE:{rules:[64,65,66],inclusive:!1},NOTE_TEXT:{rules:[69,70],inclusive:!1},NOTE_ID:{rules:[68],inclusive:!1},NOTE:{rules:[61,62,63],inclusive:!1},STYLEDEF_STYLEOPTS:{rules:[],inclusive:!1},STYLEDEF_STYLES:{rules:[34],inclusive:!1},STYLE_IDS:{rules:[],inclusive:!1},STYLE:{rules:[33],inclusive:!1},CLASS_STYLE:{rules:[31],inclusive:!1},CLASS:{rules:[30],inclusive:!1},CLASSDEFID:{rules:[28],inclusive:!1},CLASSDEF:{rules:[26,27],inclusive:!1},acc_descr_multiline:{rules:[23,24],inclusive:!1},acc_descr:{rules:[21],inclusive:!1},acc_title:{rules:[19],inclusive:!1},SCALE:{rules:[16,17,36,37],inclusive:!1},ALIAS:{rules:[],inclusive:!1},STATE_ID:{rules:[51],inclusive:!1},STATE_STRING:{rules:[52,53],inclusive:!1},FORK_STATE:{rules:[],inclusive:!1},STATE:{rules:[12,13,39,40,41,42,43,44,49,50,54,55,56],inclusive:!1},ID:{rules:[12,13],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,18,20,22,25,29,32,35,38,56,60,71,72,73,74,75,76,77,79,80,81],inclusive:!0}}}}();function at(){this.yy={}}return yt.lexer=Kt,d(at,"Parser"),at.prototype=yt,yt.Parser=at,new at}();mt.parser=mt;var he=mt,J="state",K="root",St="relation",Q="default",vt="divider",It="fill:none",At="fill: #333",Lt="text",wt="normal",Tt="rect",_t="rectWithTitle",Nt="divider",Ot="roundedWithTitle",Z="statediagram",de=`${Z}-state`,Rt="transition",ue=`${Rt} note-edge`,pe=`${Z}-note`,ye=`${Z}-cluster`,ge=`${Z}-cluster-alt`,Bt="parent",Ft="note",bt="----",fe=`${bt}${Ft}`,Pt=`${bt}${Bt}`,Yt=d((e,t="TB")=>{if(!e.doc)return t;let s=t;for(const r of e.doc)r.stmt==="dir"&&(s=r.value);return s},"getDir"),me={getClasses:d(function(e,t){return t.db.getClasses()},"getClasses"),draw:d(async function(e,t,s,r){k.info("REF0:"),k.info("Drawing state diagram (v2)",t);const{securityLevel:i,state:o,layout:l}=R();r.db.extract(r.db.getRootDocV2());const g=r.db.getData(),u=Qt(t,i);g.type=r.type,g.layoutAlgorithm=l,g.nodeSpacing=(o==null?void 0:o.nodeSpacing)||50,g.rankSpacing=(o==null?void 0:o.rankSpacing)||50,g.markers=["barb"],g.diagramId=t,await te(g,u);try{(typeof r.db.getLinks=="function"?r.db.getLinks():new Map).forEach((m,S)=>{var h;const C=typeof S=="string"?S:typeof(S==null?void 0:S.id)=="string"?S.id:"";if(!C)return void k.warn("\u26A0\uFE0F Invalid or missing stateId from key:",JSON.stringify(S));const v=(h=u.node())==null?void 0:h.querySelectorAll("g");let D;if(v==null||v.forEach(E=>{var O;((O=E.textContent)==null?void 0:O.trim())===C&&(D=E)}),!D)return void k.warn("\u26A0\uFE0F Could not find node matching text:",C);const w=D.parentNode;if(!w)return void k.warn("\u26A0\uFE0F Node has no parent, cannot wrap:",C);const x=document.createElementNS("http://www.w3.org/2000/svg","a"),L=m.url.replace(/^"+|"+$/g,"");if(x.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href",L),x.setAttribute("target","_blank"),m.tooltip){const E=m.tooltip.replace(/^"+|"+$/g,"");x.setAttribute("title",E)}w.replaceChild(x,D),x.appendChild(D),k.info("\u{1F517} Wrapped node in <a> tag for:",C,m.url)})}catch(m){k.error("\u274C Error injecting clickable links:",m)}ee.insertTitle(u,"statediagramTitleText",(o==null?void 0:o.titleTopMargin)??25,r.db.getDiagramTitle()),Zt(u,8,Z,(o==null?void 0:o.useMaxWidth)??!0)},"draw"),getDir:Yt},ht=new Map,G=0;function dt(e="",t=0,s="",r=bt){return`state-${e}${s!==null&&s.length>0?`${r}${s}`:""}-${t}`}d(dt,"stateDomId");var Se=d((e,t,s,r,i,o,l,g)=>{k.trace("items",t),t.forEach(u=>{switch(u.stmt){case J:case Q:ut(e,u,s,r,i,o,l,g);break;case St:{ut(e,u.state1,s,r,i,o,l,g),ut(e,u.state2,s,r,i,o,l,g);const m={id:"edge"+G,start:u.state1.id,end:u.state2.id,arrowhead:"normal",arrowTypeEnd:"arrow_barb",style:It,labelStyle:"",label:U.sanitizeText(u.description??"",R()),arrowheadStyle:At,labelpos:"c",labelType:Lt,thickness:wt,classes:Rt,look:l};i.push(m),G++}}})},"setupDoc"),Gt=d((e,t="TB")=>{let s=t;if(e.doc)for(const r of e.doc)r.stmt==="dir"&&(s=r.value);return s},"getDir");function tt(e,t,s){if(!t.id||t.id==="</join></fork>"||t.id==="</choice>")return;t.cssClasses&&(Array.isArray(t.cssCompiledStyles)||(t.cssCompiledStyles=[]),t.cssClasses.split(" ").forEach(i=>{const o=s.get(i);o&&(t.cssCompiledStyles=[...t.cssCompiledStyles??[],...o.styles])}));const r=e.find(i=>i.id===t.id);r?Object.assign(r,t):e.push(t)}function jt(e){var t;return((t=e==null?void 0:e.classes)==null?void 0:t.join(" "))??""}function zt(e){return(e==null?void 0:e.styles)??[]}d(tt,"insertOrUpdateNode"),d(jt,"getClassesFromDbInfo"),d(zt,"getStylesFromDbInfo");var ut=d((e,t,s,r,i,o,l,g)=>{var D,w,x;const u=t.id,m=s.get(u),S=jt(m),C=zt(m),v=R();if(k.info("dataFetcher parsedItem",t,m,C),u!=="root"){let L=Tt;t.start===!0?L="stateStart":t.start===!1&&(L="stateEnd"),t.type!==Q&&(L=t.type),ht.get(u)||ht.set(u,{id:u,shape:L,description:U.sanitizeText(u,v),cssClasses:`${S} ${de}`,cssStyles:C});const h=ht.get(u);t.description&&(Array.isArray(h.description)?(h.shape=_t,h.description.push(t.description)):(D=h.description)!=null&&D.length&&h.description.length>0?(h.shape=_t,h.description===u?h.description=[t.description]:h.description=[h.description,t.description]):(h.shape=Tt,h.description=t.description),h.description=U.sanitizeTextOrArray(h.description,v)),((w=h.description)==null?void 0:w.length)===1&&h.shape===_t&&(h.type==="group"?h.shape=Ot:h.shape=Tt),!h.type&&t.doc&&(k.info("Setting cluster for XCX",u,Gt(t)),h.type="group",h.isGroup=!0,h.dir=Gt(t),h.shape=t.type===vt?Nt:Ot,h.cssClasses=`${h.cssClasses} ${ye} ${o?ge:""}`);const E={labelStyle:"",shape:h.shape,label:h.description,cssClasses:h.cssClasses,cssCompiledStyles:[],cssStyles:h.cssStyles,id:u,dir:h.dir,domId:dt(u,G),type:h.type,isGroup:h.type==="group",padding:8,rx:10,ry:10,look:l};if(E.shape===Nt&&(E.label=""),e&&e.id!=="root"&&(k.trace("Setting node ",u," to be child of its parent ",e.id),E.parentId=e.id),E.centerLabel=!0,t.note){const N={labelStyle:"",shape:"note",label:t.note.text,cssClasses:pe,cssStyles:[],cssCompiledStyles:[],id:u+fe+"-"+G,domId:dt(u,G,Ft),type:h.type,isGroup:h.type==="group",padding:(x=v.flowchart)==null?void 0:x.padding,look:l,position:t.note.position},O=u+Pt,j={labelStyle:"",shape:"noteGroup",label:t.note.text,cssClasses:h.cssClasses,cssStyles:[],id:u+Pt,domId:dt(u,G,Bt),type:"group",isGroup:!0,padding:16,look:l,position:t.note.position};G++,j.id=O,N.parentId=O,tt(r,j,g),tt(r,N,g),tt(r,E,g);let P=u,Y=N.id;t.note.position==="left of"&&(P=N.id,Y=u),i.push({id:P+"-"+Y,start:P,end:Y,arrowhead:"none",arrowTypeEnd:"",style:It,labelStyle:"",classes:ue,arrowheadStyle:At,labelpos:"c",labelType:Lt,thickness:wt,look:l})}else tt(r,E,g)}t.doc&&(k.trace("Adding nodes children "),Se(t,t.doc,s,r,i,!o,l,g))},"dataFetcher"),Te=d(()=>{ht.clear(),G=0},"reset"),kt="[*]",Mt="start",Ut="[*]",Wt="end",Ht="color",Xt="fill",_e="bgFill",be=",",Vt=d(()=>new Map,"newClassesList"),Jt=d(()=>({relations:[],states:new Map,documents:{}}),"newDoc"),pt=d(e=>JSON.parse(JSON.stringify(e)),"clone"),ke=(W=class{constructor(t){this.version=t,this.nodes=[],this.edges=[],this.rootDoc=[],this.classes=Vt(),this.documents={root:Jt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.dividerCnt=0,this.links=new Map,this.getAccTitle=se,this.setAccTitle=ie,this.getAccDescription=ne,this.setAccDescription=re,this.setDiagramTitle=ae,this.getDiagramTitle=oe,this.clear(),this.setRootDoc=this.setRootDoc.bind(this),this.getDividerId=this.getDividerId.bind(this),this.setDirection=this.setDirection.bind(this),this.trimColon=this.trimColon.bind(this)}extract(t){this.clear(!0);for(const i of Array.isArray(t)?t:t.doc)switch(i.stmt){case J:this.addState(i.id.trim(),i.type,i.doc,i.description,i.note);break;case St:this.addRelation(i.state1,i.state2,i.description);break;case"classDef":this.addStyleClass(i.id.trim(),i.classes);break;case"style":this.handleStyleDef(i);break;case"applyClass":this.setCssClass(i.id.trim(),i.styleClass);break;case"click":this.addLink(i.id,i.url,i.tooltip)}const s=this.getStates(),r=R();Te(),ut(void 0,this.getRootDocV2(),s,this.nodes,this.edges,!0,r.look,this.classes);for(const i of this.nodes)if(Array.isArray(i.label)){if(i.description=i.label.slice(1),i.isGroup&&i.description.length>0)throw new Error(`Group nodes can only have label. Remove the additional description for node [${i.id}]`);i.label=i.label[0]}}handleStyleDef(t){const s=t.id.trim().split(","),r=t.styleClass.split(",");for(const i of s){let o=this.getState(i);if(!o){const l=i.trim();this.addState(l),o=this.getState(l)}o&&(o.styles=r.map(l=>{var g;return(g=l.replace(/;/g,""))==null?void 0:g.trim()}))}}setRootDoc(t){k.info("Setting root doc",t),this.rootDoc=t,this.version===1?this.extract(t):this.extract(this.getRootDocV2())}docTranslator(t,s,r){if(s.stmt===St)return this.docTranslator(t,s.state1,!0),void this.docTranslator(t,s.state2,!1);if(s.stmt===J&&(s.id===kt?(s.id=t.id+(r?"_start":"_end"),s.start=r):s.id=s.id.trim()),s.stmt!==K&&s.stmt!==J||!s.doc)return;const i=[];let o=[];for(const l of s.doc)if(l.type===vt){const g=pt(l);g.doc=pt(o),i.push(g),o=[]}else o.push(l);if(i.length>0&&o.length>0){const l={stmt:J,id:ce(),type:"divider",doc:pt(o)};i.push(pt(l)),s.doc=i}s.doc.forEach(l=>this.docTranslator(s,l,!0))}getRootDocV2(){return this.docTranslator({id:K,stmt:K},{id:K,stmt:K,doc:this.rootDoc},!0),{id:K,doc:this.rootDoc}}addState(t,s=Q,r=void 0,i=void 0,o=void 0,l=void 0,g=void 0,u=void 0){const m=t==null?void 0:t.trim();if(this.currentDocument.states.has(m)){const S=this.currentDocument.states.get(m);if(!S)throw new Error(`State not found: ${m}`);S.doc||(S.doc=r),S.type||(S.type=s)}else k.info("Adding state ",m,i),this.currentDocument.states.set(m,{stmt:J,id:m,descriptions:[],type:s,doc:r,note:o,classes:[],styles:[],textStyles:[]});if(i&&(k.info("Setting state description",m,i),(Array.isArray(i)?i:[i]).forEach(S=>this.addDescription(m,S.trim()))),o){const S=this.currentDocument.states.get(m);if(!S)throw new Error(`State not found: ${m}`);S.note=o,S.note.text=U.sanitizeText(S.note.text,R())}l&&(k.info("Setting state classes",m,l),(Array.isArray(l)?l:[l]).forEach(S=>this.setCssClass(m,S.trim()))),g&&(k.info("Setting state styles",m,g),(Array.isArray(g)?g:[g]).forEach(S=>this.setStyle(m,S.trim()))),u&&(k.info("Setting state styles",m,g),(Array.isArray(u)?u:[u]).forEach(S=>this.setTextStyle(m,S.trim())))}clear(t){this.nodes=[],this.edges=[],this.documents={root:Jt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.classes=Vt(),t||(this.links=new Map,le())}getState(t){return this.currentDocument.states.get(t)}getStates(){return this.currentDocument.states}logDocuments(){k.info("Documents = ",this.documents)}getRelations(){return this.currentDocument.relations}addLink(t,s,r){this.links.set(t,{url:s,tooltip:r}),k.warn("Adding link",t,s,r)}getLinks(){return this.links}startIdIfNeeded(t=""){return t===kt?(this.startEndCount++,`${Mt}${this.startEndCount}`):t}startTypeIfNeeded(t="",s=Q){return t===kt?Mt:s}endIdIfNeeded(t=""){return t===Ut?(this.startEndCount++,`${Wt}${this.startEndCount}`):t}endTypeIfNeeded(t="",s=Q){return t===Ut?Wt:s}addRelationObjs(t,s,r=""){const i=this.startIdIfNeeded(t.id.trim()),o=this.startTypeIfNeeded(t.id.trim(),t.type),l=this.startIdIfNeeded(s.id.trim()),g=this.startTypeIfNeeded(s.id.trim(),s.type);this.addState(i,o,t.doc,t.description,t.note,t.classes,t.styles,t.textStyles),this.addState(l,g,s.doc,s.description,s.note,s.classes,s.styles,s.textStyles),this.currentDocument.relations.push({id1:i,id2:l,relationTitle:U.sanitizeText(r,R())})}addRelation(t,s,r){if(typeof t=="object"&&typeof s=="object")this.addRelationObjs(t,s,r);else if(typeof t=="string"&&typeof s=="string"){const i=this.startIdIfNeeded(t.trim()),o=this.startTypeIfNeeded(t),l=this.endIdIfNeeded(s.trim()),g=this.endTypeIfNeeded(s);this.addState(i,o),this.addState(l,g),this.currentDocument.relations.push({id1:i,id2:l,relationTitle:r?U.sanitizeText(r,R()):void 0})}}addDescription(t,s){var o;const r=this.currentDocument.states.get(t),i=s.startsWith(":")?s.replace(":","").trim():s;(o=r==null?void 0:r.descriptions)==null||o.push(U.sanitizeText(i,R()))}cleanupLabel(t){return t.startsWith(":")?t.slice(2).trim():t.trim()}getDividerId(){return this.dividerCnt++,`divider-id-${this.dividerCnt}`}addStyleClass(t,s=""){this.classes.has(t)||this.classes.set(t,{id:t,styles:[],textStyles:[]});const r=this.classes.get(t);s&&r&&s.split(be).forEach(i=>{const o=i.replace(/([^;]*);/,"$1").trim();if(RegExp(Ht).exec(i)){const l=o.replace(Xt,_e).replace(Ht,Xt);r.textStyles.push(l)}r.styles.push(o)})}getClasses(){return this.classes}setCssClass(t,s){t.split(",").forEach(r=>{var o;let i=this.getState(r);if(!i){const l=r.trim();this.addState(l),i=this.getState(l)}(o=i==null?void 0:i.classes)==null||o.push(s)})}setStyle(t,s){var r,i;(i=(r=this.getState(t))==null?void 0:r.styles)==null||i.push(s)}setTextStyle(t,s){var r,i;(i=(r=this.getState(t))==null?void 0:r.textStyles)==null||i.push(s)}getDirectionStatement(){return this.rootDoc.find(t=>t.stmt==="dir")}getDirection(){var t;return((t=this.getDirectionStatement())==null?void 0:t.value)??"TB"}setDirection(t){const s=this.getDirectionStatement();s?s.value=t:this.rootDoc.unshift({stmt:"dir",value:t})}trimColon(t){return t.startsWith(":")?t.slice(1).trim():t.trim()}getData(){const t=R();return{nodes:this.nodes,edges:this.edges,other:{},config:t,direction:Yt(this.getRootDocV2())}}getConfig(){return R().state}},d(W,"StateDB"),W.relationType={AGGREGATION:0,EXTENSION:1,COMPOSITION:2,DEPENDENCY:3},W),Ee=d(e=>`
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
`,"getStyles");export{ke as S,he as a,me as b,Ee as s};
