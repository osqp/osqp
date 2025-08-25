import{_ as l,g as le,s as de,q as ue,p as he,a as fe,b as me,c as ot,d as ye,ap as V,l as mt,j as ke,i as pe,y as ge,u as be}from"./mermaid-D59lkToe.js";import{z as gt,bR as Wt,bJ as Ot,bI as xe,cn as ve,co as Te,cp as we,cq as _e,cr as De,cs as $e,bH as Pt,bG as zt,bF as Bt,bv as Ht,bE as Nt}from"./index-D4bXoNM3.js";import{s as bt}from"./transform-B8bpuzxV.js";import{t as Ce,m as Se,a as Me,i as Ee}from"./time-1P0zCPa4.js";import{l as Ae}from"./linear-BsI0Wmp7.js";import"./step-BwsUM5iJ.js";import"./timer-BwIYMJWC.js";import"./init-DLRA0X12.js";function Le(e){return e}var Gt=1e-6;function Ye(e){return"translate("+e+",0)"}function Ie(e){return"translate(0,"+e+")"}function Fe(e){return n=>+e(n)}function We(e,n){return n=Math.max(0,e.bandwidth()-2*n)/2,e.round()&&(n=Math.round(n)),s=>+e(s)+n}function Oe(){return!this.__axis}function jt(e,n){var s=[],r=null,i=null,c=6,u=6,A=3,v=typeof window<"u"&&window.devicePixelRatio>1?0:.5,$=e===1||e===4?-1:1,L=e===4||e===2?"x":"y",F=e===1||e===3?Ye:Ie;function S(D){var g=r??(n.ticks?n.ticks.apply(n,s):n.domain()),k=i??(n.tickFormat?n.tickFormat.apply(n,s):Le),M=Math.max(c,0)+A,E=n.range(),P=+E[0]+v,z=+E[E.length-1]+v,j=(n.bandwidth?We:Fe)(n.copy(),v),B=D.selection?D.selection():D,W=B.selectAll(".domain").data([null]),I=B.selectAll(".tick").data(g,n).order(),p=I.exit(),w=I.enter().append("g").attr("class","tick"),x=I.select("line"),b=I.select("text");W=W.merge(W.enter().insert("path",".tick").attr("class","domain").attr("stroke","currentColor")),I=I.merge(w),x=x.merge(w.append("line").attr("stroke","currentColor").attr(L+"2",$*c)),b=b.merge(w.append("text").attr("fill","currentColor").attr(L,$*M).attr("dy",e===1?"0em":e===3?"0.71em":"0.32em")),D!==B&&(W=W.transition(D),I=I.transition(D),x=x.transition(D),b=b.transition(D),p=p.transition(D).attr("opacity",Gt).attr("transform",function(y){return isFinite(y=j(y))?F(y+v):this.getAttribute("transform")}),w.attr("opacity",Gt).attr("transform",function(y){var o=this.parentNode.__axis;return F((o&&isFinite(o=o(y))?o:j(y))+v)})),p.remove(),W.attr("d",e===4||e===2?u?"M"+$*u+","+P+"H"+v+"V"+z+"H"+$*u:"M"+v+","+P+"V"+z:u?"M"+P+","+$*u+"V"+v+"H"+z+"V"+$*u:"M"+P+","+v+"H"+z),I.attr("opacity",1).attr("transform",function(y){return F(j(y)+v)}),x.attr(L+"2",$*c),b.attr(L,$*M).text(k),B.filter(Oe).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",e===2?"start":e===4?"end":"middle"),B.each(function(){this.__axis=j})}return S.scale=function(D){return arguments.length?(n=D,S):n},S.ticks=function(){return s=Array.from(arguments),S},S.tickArguments=function(D){return arguments.length?(s=D==null?[]:Array.from(D),S):s.slice()},S.tickValues=function(D){return arguments.length?(r=D==null?null:Array.from(D),S):r&&r.slice()},S.tickFormat=function(D){return arguments.length?(i=D,S):i},S.tickSize=function(D){return arguments.length?(c=u=+D,S):c},S.tickSizeInner=function(D){return arguments.length?(c=+D,S):c},S.tickSizeOuter=function(D){return arguments.length?(u=+D,S):u},S.tickPadding=function(D){return arguments.length?(A=+D,S):A},S.offset=function(D){return arguments.length?(v=+D,S):v},S}var Vt,yt,Rt={exports:{}};const Pe=gt(Vt?Rt.exports:(Vt=1,Rt.exports=(yt="day",function(e,n,s){var r=function(u){return u.add(4-u.isoWeekday(),yt)},i=n.prototype;i.isoWeekYear=function(){return r(this).year()},i.isoWeek=function(u){if(!this.$utils().u(u))return this.add(7*(u-this.isoWeek()),yt);var A,v,$,L=r(this),F=(A=this.isoWeekYear(),$=4-(v=(this.$u?s.utc:s)().year(A).startOf("year")).isoWeekday(),v.isoWeekday()>4&&($+=7),v.add($,yt));return L.diff(F,"week")+1},i.isoWeekday=function(u){return this.$utils().u(u)?this.day()||7:this.day(this.day()%7?u:u-7)};var c=i.startOf;i.startOf=function(u,A){var v=this.$utils(),$=!!v.u(A)||A;return v.p(u)==="isoweek"?$?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):c.bind(this)(u,A)}})));var Zt,qt={exports:{}},ze=(Zt||(Zt=1,qt.exports=function(){var e={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},n=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,s=/\d/,r=/\d\d/,i=/\d\d?/,c=/\d*[^-_:/,()\s\d]+/,u={},A=function(g){return(g=+g)+(g>68?1900:2e3)},v=function(g){return function(k){this[g]=+k}},$=[/[+-]\d\d:?(\d\d)?|Z/,function(g){(this.zone||(this.zone={})).offset=function(k){if(!k||k==="Z")return 0;var M=k.match(/([+-]|\d\d)/g),E=60*M[1]+(+M[2]||0);return E===0?0:M[0]==="+"?-E:E}(g)}],L=function(g){var k=u[g];return k&&(k.indexOf?k:k.s.concat(k.f))},F=function(g,k){var M,E=u.meridiem;if(E){for(var P=1;P<=24;P+=1)if(g.indexOf(E(P,0,k))>-1){M=P>12;break}}else M=g===(k?"pm":"PM");return M},S={A:[c,function(g){this.afternoon=F(g,!1)}],a:[c,function(g){this.afternoon=F(g,!0)}],Q:[s,function(g){this.month=3*(g-1)+1}],S:[s,function(g){this.milliseconds=100*+g}],SS:[r,function(g){this.milliseconds=10*+g}],SSS:[/\d{3}/,function(g){this.milliseconds=+g}],s:[i,v("seconds")],ss:[i,v("seconds")],m:[i,v("minutes")],mm:[i,v("minutes")],H:[i,v("hours")],h:[i,v("hours")],HH:[i,v("hours")],hh:[i,v("hours")],D:[i,v("day")],DD:[r,v("day")],Do:[c,function(g){var k=u.ordinal,M=g.match(/\d+/);if(this.day=M[0],k)for(var E=1;E<=31;E+=1)k(E).replace(/\[|\]/g,"")===g&&(this.day=E)}],w:[i,v("week")],ww:[r,v("week")],M:[i,v("month")],MM:[r,v("month")],MMM:[c,function(g){var k=L("months"),M=(L("monthsShort")||k.map(function(E){return E.slice(0,3)})).indexOf(g)+1;if(M<1)throw new Error;this.month=M%12||M}],MMMM:[c,function(g){var k=L("months").indexOf(g)+1;if(k<1)throw new Error;this.month=k%12||k}],Y:[/[+-]?\d+/,v("year")],YY:[r,function(g){this.year=A(g)}],YYYY:[/\d{4}/,v("year")],Z:$,ZZ:$};function D(g){var k,M;k=g,M=u&&u.formats;for(var E=(g=k.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,function(p,w,x){var b=x&&x.toUpperCase();return w||M[x]||e[x]||M[b].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,function(y,o,m){return o||m.slice(1)})})).match(n),P=E.length,z=0;z<P;z+=1){var j=E[z],B=S[j],W=B&&B[0],I=B&&B[1];E[z]=I?{regex:W,parser:I}:j.replace(/^\[|\]$/g,"")}return function(p){for(var w={},x=0,b=0;x<P;x+=1){var y=E[x];if(typeof y=="string")b+=y.length;else{var o=y.regex,m=y.parser,h=p.slice(b),d=o.exec(h)[0];m.call(w,d),p=p.replace(d,"")}}return function(T){var t=T.afternoon;if(t!==void 0){var f=T.hours;t?f<12&&(T.hours+=12):f===12&&(T.hours=0),delete T.afternoon}}(w),w}}return function(g,k,M){M.p.customParseFormat=!0,g&&g.parseTwoDigitYear&&(A=g.parseTwoDigitYear);var E=k.prototype,P=E.parse;E.parse=function(z){var j=z.date,B=z.utc,W=z.args;this.$u=B;var I=W[1];if(typeof I=="string"){var p=W[2]===!0,w=W[3]===!0,x=p||w,b=W[2];w&&(b=W[2]),u=this.$locale(),!p&&b&&(u=M.Ls[b]),this.$d=function(h,d,T,t){try{if(["x","X"].indexOf(d)>-1)return new Date((d==="X"?1e3:1)*h);var f=D(d)(h),a=f.year,_=f.month,Y=f.day,O=f.hours,C=f.minutes,q=f.seconds,tt=f.milliseconds,rt=f.zone,ft=f.week,st=new Date,N=Y||(a||_?1:st.getDate()),U=a||st.getFullYear(),G=0;a&&!_||(G=_>0?_-1:st.getMonth());var et,X=O||0,R=C||0,at=q||0,J=tt||0;return rt?new Date(Date.UTC(U,G,N,X,R,at,J+60*rt.offset*1e3)):T?new Date(Date.UTC(U,G,N,X,R,at,J)):(et=new Date(U,G,N,X,R,at,J),ft&&(et=t(et).week(ft).toDate()),et)}catch{return new Date("")}}(j,I,B,M),this.init(),b&&b!==!0&&(this.$L=this.locale(b).$L),x&&j!=this.format(I)&&(this.$d=new Date("")),u={}}else if(I instanceof Array)for(var y=I.length,o=1;o<=y;o+=1){W[1]=I[o-1];var m=M.apply(this,W);if(m.isValid()){this.$d=m.$d,this.$L=m.$L,this.init();break}o===y&&(this.$d=new Date(""))}else P.call(this,z)}}}()),qt.exports);const Be=gt(ze);var Ut,Qt={exports:{}},He=(Ut||(Ut=1,Qt.exports=function(e,n){var s=n.prototype,r=s.format;s.format=function(i){var c=this,u=this.$locale();if(!this.isValid())return r.bind(this)(i);var A=this.$utils(),v=(i||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,function($){switch($){case"Q":return Math.ceil((c.$M+1)/3);case"Do":return u.ordinal(c.$D);case"gggg":return c.weekYear();case"GGGG":return c.isoWeekYear();case"wo":return u.ordinal(c.week(),"W");case"w":case"ww":return A.s(c.week(),$==="w"?1:2,"0");case"W":case"WW":return A.s(c.isoWeek(),$==="W"?1:2,"0");case"k":case"kk":return A.s(String(c.$H===0?24:c.$H),$==="k"?1:2,"0");case"X":return Math.floor(c.$d.getTime()/1e3);case"x":return c.$d.getTime();case"z":return"["+c.offsetName()+"]";case"zzz":return"["+c.offsetName("long")+"]";default:return $}});return r.bind(this)(v)}}),Qt.exports);const Ne=gt(He);var xt=function(){var e=l(function(o,m,h,d){for(h=h||{},d=o.length;d--;h[o[d]]=m);return h},"o"),n=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],s=[1,26],r=[1,27],i=[1,28],c=[1,29],u=[1,30],A=[1,31],v=[1,32],$=[1,33],L=[1,34],F=[1,9],S=[1,10],D=[1,11],g=[1,12],k=[1,13],M=[1,14],E=[1,15],P=[1,16],z=[1,19],j=[1,20],B=[1,21],W=[1,22],I=[1,23],p=[1,25],w=[1,35],x={trace:l(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:l(function(o,m,h,d,T,t,f){var a=t.length-1;switch(T){case 1:return t[a-1];case 2:case 6:case 7:this.$=[];break;case 3:t[a-1].push(t[a]),this.$=t[a-1];break;case 4:case 5:this.$=t[a];break;case 8:d.setWeekday("monday");break;case 9:d.setWeekday("tuesday");break;case 10:d.setWeekday("wednesday");break;case 11:d.setWeekday("thursday");break;case 12:d.setWeekday("friday");break;case 13:d.setWeekday("saturday");break;case 14:d.setWeekday("sunday");break;case 15:d.setWeekend("friday");break;case 16:d.setWeekend("saturday");break;case 17:d.setDateFormat(t[a].substr(11)),this.$=t[a].substr(11);break;case 18:d.enableInclusiveEndDates(),this.$=t[a].substr(18);break;case 19:d.TopAxis(),this.$=t[a].substr(8);break;case 20:d.setAxisFormat(t[a].substr(11)),this.$=t[a].substr(11);break;case 21:d.setTickInterval(t[a].substr(13)),this.$=t[a].substr(13);break;case 22:d.setExcludes(t[a].substr(9)),this.$=t[a].substr(9);break;case 23:d.setIncludes(t[a].substr(9)),this.$=t[a].substr(9);break;case 24:d.setTodayMarker(t[a].substr(12)),this.$=t[a].substr(12);break;case 27:d.setDiagramTitle(t[a].substr(6)),this.$=t[a].substr(6);break;case 28:this.$=t[a].trim(),d.setAccTitle(this.$);break;case 29:case 30:this.$=t[a].trim(),d.setAccDescription(this.$);break;case 31:d.addSection(t[a].substr(8)),this.$=t[a].substr(8);break;case 33:d.addTask(t[a-1],t[a]),this.$="task";break;case 34:this.$=t[a-1],d.setClickEvent(t[a-1],t[a],null);break;case 35:this.$=t[a-2],d.setClickEvent(t[a-2],t[a-1],t[a]);break;case 36:this.$=t[a-2],d.setClickEvent(t[a-2],t[a-1],null),d.setLink(t[a-2],t[a]);break;case 37:this.$=t[a-3],d.setClickEvent(t[a-3],t[a-2],t[a-1]),d.setLink(t[a-3],t[a]);break;case 38:this.$=t[a-2],d.setClickEvent(t[a-2],t[a],null),d.setLink(t[a-2],t[a-1]);break;case 39:this.$=t[a-3],d.setClickEvent(t[a-3],t[a-1],t[a]),d.setLink(t[a-3],t[a-2]);break;case 40:this.$=t[a-1],d.setLink(t[a-1],t[a]);break;case 41:case 47:this.$=t[a-1]+" "+t[a];break;case 42:case 43:case 45:this.$=t[a-2]+" "+t[a-1]+" "+t[a];break;case 44:case 46:this.$=t[a-3]+" "+t[a-2]+" "+t[a-1]+" "+t[a]}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},e(n,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:s,13:r,14:i,15:c,16:u,17:A,18:v,19:18,20:$,21:L,22:F,23:S,24:D,25:g,26:k,27:M,28:E,29:P,30:z,31:j,33:B,35:W,36:I,37:24,38:p,40:w},e(n,[2,7],{1:[2,1]}),e(n,[2,3]),{9:36,11:17,12:s,13:r,14:i,15:c,16:u,17:A,18:v,19:18,20:$,21:L,22:F,23:S,24:D,25:g,26:k,27:M,28:E,29:P,30:z,31:j,33:B,35:W,36:I,37:24,38:p,40:w},e(n,[2,5]),e(n,[2,6]),e(n,[2,17]),e(n,[2,18]),e(n,[2,19]),e(n,[2,20]),e(n,[2,21]),e(n,[2,22]),e(n,[2,23]),e(n,[2,24]),e(n,[2,25]),e(n,[2,26]),e(n,[2,27]),{32:[1,37]},{34:[1,38]},e(n,[2,30]),e(n,[2,31]),e(n,[2,32]),{39:[1,39]},e(n,[2,8]),e(n,[2,9]),e(n,[2,10]),e(n,[2,11]),e(n,[2,12]),e(n,[2,13]),e(n,[2,14]),e(n,[2,15]),e(n,[2,16]),{41:[1,40],43:[1,41]},e(n,[2,4]),e(n,[2,28]),e(n,[2,29]),e(n,[2,33]),e(n,[2,34],{42:[1,42],43:[1,43]}),e(n,[2,40],{41:[1,44]}),e(n,[2,35],{43:[1,45]}),e(n,[2,36]),e(n,[2,38],{42:[1,46]}),e(n,[2,37]),e(n,[2,39])],defaultActions:{},parseError:l(function(o,m){if(!m.recoverable){var h=new Error(o);throw h.hash=m,h}this.trace(o)},"parseError"),parse:l(function(o){var m=this,h=[0],d=[],T=[null],t=[],f=this.table,a="",_=0,Y=0,O=t.slice.call(arguments,1),C=Object.create(this.lexer),q={yy:{}};for(var tt in this.yy)Object.prototype.hasOwnProperty.call(this.yy,tt)&&(q.yy[tt]=this.yy[tt]);C.setInput(o,q.yy),q.yy.lexer=C,q.yy.parser=this,C.yylloc===void 0&&(C.yylloc={});var rt=C.yylloc;t.push(rt);var ft=C.options&&C.options.ranges;function st(){var Z;return typeof(Z=d.pop()||C.lex()||1)!="number"&&(Z instanceof Array&&(Z=(d=Z).pop()),Z=m.symbols_[Z]||Z),Z}typeof q.yy.parseError=="function"?this.parseError=q.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError,l(function(Z){h.length=h.length-2*Z,T.length=T.length-Z,t.length=t.length-Z},"popStack"),l(st,"lex");for(var N,U,G,et,X,R,at,J,nt={};;){if(U=h[h.length-1],this.defaultActions[U]?G=this.defaultActions[U]:(N==null&&(N=st()),G=f[U]&&f[U][N]),G===void 0||!G.length||!G[0]){var Ft="";for(X in J=[],f[U])this.terminals_[X]&&X>2&&J.push("'"+this.terminals_[X]+"'");Ft=C.showPosition?"Parse error on line "+(_+1)+`:
`+C.showPosition()+`
Expecting `+J.join(", ")+", got '"+(this.terminals_[N]||N)+"'":"Parse error on line "+(_+1)+": Unexpected "+(N==1?"end of input":"'"+(this.terminals_[N]||N)+"'"),this.parseError(Ft,{text:C.match,token:this.terminals_[N]||N,line:C.yylineno,loc:rt,expected:J})}if(G[0]instanceof Array&&G.length>1)throw new Error("Parse Error: multiple actions possible at state: "+U+", token: "+N);switch(G[0]){case 1:h.push(N),T.push(C.yytext),t.push(C.yylloc),h.push(G[1]),N=null,Y=C.yyleng,a=C.yytext,_=C.yylineno,rt=C.yylloc;break;case 2:if(R=this.productions_[G[1]][1],nt.$=T[T.length-R],nt._$={first_line:t[t.length-(R||1)].first_line,last_line:t[t.length-1].last_line,first_column:t[t.length-(R||1)].first_column,last_column:t[t.length-1].last_column},ft&&(nt._$.range=[t[t.length-(R||1)].range[0],t[t.length-1].range[1]]),(et=this.performAction.apply(nt,[a,Y,_,q.yy,G[1],T,t].concat(O)))!==void 0)return et;R&&(h=h.slice(0,-1*R*2),T=T.slice(0,-1*R),t=t.slice(0,-1*R)),h.push(this.productions_[G[1]][0]),T.push(nt.$),t.push(nt._$),at=f[h[h.length-2]][h[h.length-1]],h.push(at);break;case 3:return!0}}return!0},"parse")},b=function(){return{EOF:1,parseError:l(function(o,m){if(!this.yy.parser)throw new Error(o);this.yy.parser.parseError(o,m)},"parseError"),setInput:l(function(o,m){return this.yy=m||this.yy||{},this._input=o,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:l(function(){var o=this._input[0];return this.yytext+=o,this.yyleng++,this.offset++,this.match+=o,this.matched+=o,o.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),o},"input"),unput:l(function(o){var m=o.length,h=o.split(/(?:\r\n?|\n)/g);this._input=o+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-m),this.offset-=m;var d=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var T=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===d.length?this.yylloc.first_column:0)+d[d.length-h.length].length-h[0].length:this.yylloc.first_column-m},this.options.ranges&&(this.yylloc.range=[T[0],T[0]+this.yyleng-m]),this.yyleng=this.yytext.length,this},"unput"),more:l(function(){return this._more=!0,this},"more"),reject:l(function(){return this.options.backtrack_lexer?(this._backtrack=!0,this):this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"reject"),less:l(function(o){this.unput(this.match.slice(o))},"less"),pastInput:l(function(){var o=this.matched.substr(0,this.matched.length-this.match.length);return(o.length>20?"...":"")+o.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:l(function(){var o=this.match;return o.length<20&&(o+=this._input.substr(0,20-o.length)),(o.substr(0,20)+(o.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:l(function(){var o=this.pastInput(),m=new Array(o.length+1).join("-");return o+this.upcomingInput()+`
`+m+"^"},"showPosition"),test_match:l(function(o,m){var h,d,T;if(this.options.backtrack_lexer&&(T={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(T.yylloc.range=this.yylloc.range.slice(0))),(d=o[0].match(/(?:\r\n?|\n).*/g))&&(this.yylineno+=d.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:d?d[d.length-1].length-d[d.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+o[0].length},this.yytext+=o[0],this.match+=o[0],this.matches=o,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(o[0].length),this.matched+=o[0],h=this.performAction.call(this,this.yy,this,m,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var t in T)this[t]=T[t];return!1}return!1},"test_match"),next:l(function(){if(this.done)return this.EOF;var o,m,h,d;this._input||(this.done=!0),this._more||(this.yytext="",this.match="");for(var T=this._currentRules(),t=0;t<T.length;t++)if((h=this._input.match(this.rules[T[t]]))&&(!m||h[0].length>m[0].length)){if(m=h,d=t,this.options.backtrack_lexer){if((o=this.test_match(h,T[t]))!==!1)return o;if(this._backtrack){m=!1;continue}return!1}if(!this.options.flex)break}return m?(o=this.test_match(m,T[d]))!==!1&&o:this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:l(function(){var o=this.next();return o||this.lex()},"lex"),begin:l(function(o){this.conditionStack.push(o)},"begin"),popState:l(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:l(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:l(function(o){return(o=this.conditionStack.length-1-Math.abs(o||0))>=0?this.conditionStack[o]:"INITIAL"},"topState"),pushState:l(function(o){this.begin(o)},"pushState"),stateStackSize:l(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:l(function(o,m,h,d){switch(h){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:case 15:case 18:case 21:case 24:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:case 9:case 10:case 12:case 13:break;case 11:return 10;case 14:this.begin("href");break;case 16:return 43;case 17:this.begin("callbackname");break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 22:return 42;case 23:this.begin("click");break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}}}();function y(){this.yy={}}return x.lexer=b,l(y,"Parser"),y.prototype=x,x.Parser=y,new y}();xt.parser=xt;var Ge=xt;V.extend(Pe),V.extend(Be),V.extend(Ne);var vt,kt,Xt={friday:5,saturday:6},Q="",Tt="",wt=void 0,_t="",dt=[],ut=[],Dt=new Map,$t=[],Ct=[],ct="",St="",Jt=["active","done","crit","milestone","vert"],Mt=[],ht=!1,Et=!1,At="sunday",pt="saturday",Lt=0,je=l(function(){$t=[],Ct=[],ct="",Mt=[],ie=0,vt=void 0,kt=void 0,H=[],Q="",Tt="",St="",wt=void 0,_t="",dt=[],ut=[],ht=!1,Et=!1,Lt=0,Dt=new Map,ge(),At="sunday",pt="saturday"},"clear"),Ve=l(function(e){Tt=e},"setAxisFormat"),Re=l(function(){return Tt},"getAxisFormat"),Ze=l(function(e){wt=e},"setTickInterval"),qe=l(function(){return wt},"getTickInterval"),Ue=l(function(e){_t=e},"setTodayMarker"),Qe=l(function(){return _t},"getTodayMarker"),Xe=l(function(e){Q=e},"setDateFormat"),Je=l(function(){ht=!0},"enableInclusiveEndDates"),Ke=l(function(){return ht},"endDatesAreInclusive"),tn=l(function(){Et=!0},"enableTopAxis"),en=l(function(){return Et},"topAxisEnabled"),nn=l(function(e){St=e},"setDisplayMode"),rn=l(function(){return St},"getDisplayMode"),sn=l(function(){return Q},"getDateFormat"),an=l(function(e){dt=e.toLowerCase().split(/[\s,]+/)},"setIncludes"),on=l(function(){return dt},"getIncludes"),cn=l(function(e){ut=e.toLowerCase().split(/[\s,]+/)},"setExcludes"),ln=l(function(){return ut},"getExcludes"),dn=l(function(){return Dt},"getLinks"),un=l(function(e){ct=e,$t.push(e)},"addSection"),hn=l(function(){return $t},"getSections"),fn=l(function(){let e=se(),n=0;for(;!e&&n<10;)e=se(),n++;return Ct=H},"getTasks"),Kt=l(function(e,n,s,r){const i=e.format(n.trim()),c=e.format("YYYY-MM-DD");return!r.includes(i)&&!r.includes(c)&&(!(!s.includes("weekends")||e.isoWeekday()!==Xt[pt]&&e.isoWeekday()!==Xt[pt]+1)||!!s.includes(e.format("dddd").toLowerCase())||s.includes(i)||s.includes(c))},"isInvalidDate"),mn=l(function(e){At=e},"setWeekday"),yn=l(function(){return At},"getWeekday"),kn=l(function(e){pt=e},"setWeekend"),te=l(function(e,n,s,r){if(!s.length||e.manualEndTime)return;let i,c;i=e.startTime instanceof Date?V(e.startTime):V(e.startTime,n,!0),i=i.add(1,"d"),c=e.endTime instanceof Date?V(e.endTime):V(e.endTime,n,!0);const[u,A]=pn(i,c,n,s,r);e.endTime=u.toDate(),e.renderEndTime=A},"checkTaskDates"),pn=l(function(e,n,s,r,i){let c=!1,u=null;for(;e<=n;)c||(u=n.toDate()),c=Kt(e,s,r,i),c&&(n=n.add(1,"d")),e=e.add(1,"d");return[n,u]},"fixTaskDates"),Yt=l(function(e,n,s){s=s.trim();const r=/^after\s+(?<ids>[\d\w- ]+)/.exec(s);if(r!==null){let c=null;for(const A of r.groups.ids.split(" ")){let v=it(A);v!==void 0&&(!c||v.endTime>c.endTime)&&(c=v)}if(c)return c.endTime;const u=new Date;return u.setHours(0,0,0,0),u}let i=V(s,n.trim(),!0);if(i.isValid())return i.toDate();{mt.debug("Invalid date:"+s),mt.debug("With date format:"+n.trim());const c=new Date(s);if(c===void 0||isNaN(c.getTime())||c.getFullYear()<-1e4||c.getFullYear()>1e4)throw new Error("Invalid date:"+s);return c}},"getStartDate"),ee=l(function(e){const n=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(e.trim());return n!==null?[Number.parseFloat(n[1]),n[2]]:[NaN,"ms"]},"parseDuration"),ne=l(function(e,n,s,r=!1){s=s.trim();const i=/^until\s+(?<ids>[\d\w- ]+)/.exec(s);if(i!==null){let $=null;for(const F of i.groups.ids.split(" ")){let S=it(F);S!==void 0&&(!$||S.startTime<$.startTime)&&($=S)}if($)return $.startTime;const L=new Date;return L.setHours(0,0,0,0),L}let c=V(s,n.trim(),!0);if(c.isValid())return r&&(c=c.add(1,"d")),c.toDate();let u=V(e);const[A,v]=ee(s);if(!Number.isNaN(A)){const $=u.add(A,v);$.isValid()&&(u=$)}return u.toDate()},"getEndDate"),ie=0,lt=l(function(e){return e===void 0?"task"+(ie+=1):e},"parseId"),gn=l(function(e,n){let s;s=n.substr(0,1)===":"?n.substr(1,n.length):n;const r=s.split(","),i={};It(r,i,Jt);for(let u=0;u<r.length;u++)r[u]=r[u].trim();let c="";switch(r.length){case 1:i.id=lt(),i.startTime=e.endTime,c=r[0];break;case 2:i.id=lt(),i.startTime=Yt(void 0,Q,r[0]),c=r[1];break;case 3:i.id=lt(r[0]),i.startTime=Yt(void 0,Q,r[1]),c=r[2]}return c&&(i.endTime=ne(i.startTime,Q,c,ht),i.manualEndTime=V(c,"YYYY-MM-DD",!0).isValid(),te(i,Q,ut,dt)),i},"compileData"),bn=l(function(e,n){let s;s=n.substr(0,1)===":"?n.substr(1,n.length):n;const r=s.split(","),i={};It(r,i,Jt);for(let c=0;c<r.length;c++)r[c]=r[c].trim();switch(r.length){case 1:i.id=lt(),i.startTime={type:"prevTaskEnd",id:e},i.endTime={data:r[0]};break;case 2:i.id=lt(),i.startTime={type:"getStartDate",startData:r[0]},i.endTime={data:r[1]};break;case 3:i.id=lt(r[0]),i.startTime={type:"getStartDate",startData:r[1]},i.endTime={data:r[2]}}return i},"parseData"),H=[],re={},xn=l(function(e,n){const s={section:ct,type:ct,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:n},task:e,classes:[]},r=bn(kt,n);s.raw.startTime=r.startTime,s.raw.endTime=r.endTime,s.id=r.id,s.prevTaskId=kt,s.active=r.active,s.done=r.done,s.crit=r.crit,s.milestone=r.milestone,s.vert=r.vert,s.order=Lt,Lt++;const i=H.push(s);kt=s.id,re[s.id]=i-1},"addTask"),it=l(function(e){const n=re[e];return H[n]},"findTaskById"),vn=l(function(e,n){const s={section:ct,type:ct,description:e,task:e,classes:[]},r=gn(vt,n);s.startTime=r.startTime,s.endTime=r.endTime,s.id=r.id,s.active=r.active,s.done=r.done,s.crit=r.crit,s.milestone=r.milestone,s.vert=r.vert,vt=s,Ct.push(s)},"addTaskOrg"),se=l(function(){const e=l(function(s){const r=H[s];let i="";switch(H[s].raw.startTime.type){case"prevTaskEnd":{const c=it(r.prevTaskId);r.startTime=c.endTime;break}case"getStartDate":i=Yt(void 0,Q,H[s].raw.startTime.startData),i&&(H[s].startTime=i)}return H[s].startTime&&(H[s].endTime=ne(H[s].startTime,Q,H[s].raw.endTime.data,ht),H[s].endTime&&(H[s].processed=!0,H[s].manualEndTime=V(H[s].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),te(H[s],Q,ut,dt))),H[s].processed},"compileTask");let n=!0;for(const[s,r]of H.entries())e(s),n=n&&r.processed;return n},"compileTasks"),Tn=l(function(e,n){let s=n;ot().securityLevel!=="loose"&&(s=pe.sanitizeUrl(n)),e.split(",").forEach(function(r){it(r)!==void 0&&(oe(r,()=>{window.open(s,"_self")}),Dt.set(r,s))}),ae(e,"clickable")},"setLink"),ae=l(function(e,n){e.split(",").forEach(function(s){let r=it(s);r!==void 0&&r.classes.push(n)})},"setClass"),wn=l(function(e,n,s){if(ot().securityLevel!=="loose"||n===void 0)return;let r=[];if(typeof s=="string"){r=s.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let i=0;i<r.length;i++){let c=r[i].trim();c.startsWith('"')&&c.endsWith('"')&&(c=c.substr(1,c.length-2)),r[i]=c}}r.length===0&&r.push(e),it(e)!==void 0&&oe(e,()=>{be.runFunc(n,...r)})},"setClickFun"),oe=l(function(e,n){Mt.push(function(){const s=document.querySelector(`[id="${e}"]`);s!==null&&s.addEventListener("click",function(){n()})},function(){const s=document.querySelector(`[id="${e}-text"]`);s!==null&&s.addEventListener("click",function(){n()})})},"pushFun"),_n=l(function(e,n,s){e.split(",").forEach(function(r){wn(r,n,s)}),ae(e,"clickable")},"setClickEvent"),Dn=l(function(e){Mt.forEach(function(n){n(e)})},"bindFunctions"),$n={getConfig:l(()=>ot().gantt,"getConfig"),clear:je,setDateFormat:Xe,getDateFormat:sn,enableInclusiveEndDates:Je,endDatesAreInclusive:Ke,enableTopAxis:tn,topAxisEnabled:en,setAxisFormat:Ve,getAxisFormat:Re,setTickInterval:Ze,getTickInterval:qe,setTodayMarker:Ue,getTodayMarker:Qe,setAccTitle:me,getAccTitle:fe,setDiagramTitle:he,getDiagramTitle:ue,setDisplayMode:nn,getDisplayMode:rn,setAccDescription:de,getAccDescription:le,addSection:un,getSections:hn,getTasks:fn,addTask:xn,findTaskById:it,addTaskOrg:vn,setIncludes:an,getIncludes:on,setExcludes:cn,getExcludes:ln,setClickEvent:_n,setLink:Tn,getLinks:dn,bindFunctions:Dn,parseDuration:ee,isInvalidDate:Kt,setWeekday:mn,getWeekday:yn,setWeekend:kn};function It(e,n,s){let r=!0;for(;r;)r=!1,s.forEach(function(i){const c=new RegExp("^\\s*"+i+"\\s*$");e[0].match(c)&&(n[i]=!0,e.shift(1),r=!0)})}l(It,"getTaskTags");var K,Cn=l(function(){mt.debug("Something is calling, setConf, remove the call")},"setConf"),ce={monday:$e,tuesday:De,wednesday:_e,thursday:we,friday:Te,saturday:ve,sunday:xe},Sn=l((e,n)=>{let s=[...e].map(()=>-1/0),r=[...e].sort((c,u)=>c.startTime-u.startTime||c.order-u.order),i=0;for(const c of r)for(let u=0;u<s.length;u++)if(c.startTime>=s[u]){s[u]=c.endTime,c.order=u+n,u>i&&(i=u);break}return i},"getMaxIntersections"),Mn={parser:Ge,db:$n,renderer:{setConf:Cn,draw:l(function(e,n,s,r){const i=ot().gantt,c=ot().securityLevel;let u;c==="sandbox"&&(u=bt("#i"+n));const A=bt(c==="sandbox"?u.nodes()[0].contentDocument.body:"body"),v=c==="sandbox"?u.nodes()[0].contentDocument:document,$=v.getElementById(n);(K=$.parentElement.offsetWidth)===void 0&&(K=1200),i.useWidth!==void 0&&(K=i.useWidth);const L=r.db.getTasks();let F=[];for(const p of L)F.push(p.type);F=I(F);const S={};let D=2*i.topPadding;if(r.db.getDisplayMode()==="compact"||i.displayMode==="compact"){const p={};for(const x of L)p[x.section]===void 0?p[x.section]=[x]:p[x.section].push(x);let w=0;for(const x of Object.keys(p)){const b=Sn(p[x],w)+1;w+=b,D+=b*(i.barHeight+i.barGap),S[x]=b}}else{D+=L.length*(i.barHeight+i.barGap);for(const p of F)S[p]=L.filter(w=>w.type===p).length}$.setAttribute("viewBox","0 0 "+K+" "+D);const g=A.select(`[id="${n}"]`),k=Ce().domain([Se(L,function(p){return p.startTime}),Me(L,function(p){return p.endTime})]).rangeRound([0,K-i.leftPadding-i.rightPadding]);function M(p,w){const x=p.startTime,b=w.startTime;let y=0;return x>b?y=1:x<b&&(y=-1),y}function E(p,w,x){const b=i.barHeight,y=b+i.barGap,o=i.topPadding,m=i.leftPadding,h=Ae().domain([0,F.length]).range(["#00B9FA","#F95002"]).interpolate(Ee);z(y,o,m,w,x,p,r.db.getExcludes(),r.db.getIncludes()),j(m,o,w,x),P(p,y,o,m,b,h,w),B(y,o),W(m,o,w,x)}function P(p,w,x,b,y,o,m){p.sort((t,f)=>t.vert===f.vert?0:t.vert?1:-1);const h=[...new Set(p.map(t=>t.order))].map(t=>p.find(f=>f.order===t));g.append("g").selectAll("rect").data(h).enter().append("rect").attr("x",0).attr("y",function(t,f){return t.order*w+x-2}).attr("width",function(){return m-i.rightPadding/2}).attr("height",w).attr("class",function(t){for(const[f,a]of F.entries())if(t.type===a)return"section section"+f%i.numberSectionStyles;return"section section0"}).enter();const d=g.append("g").selectAll("rect").data(p).enter(),T=r.db.getLinks();if(d.append("rect").attr("id",function(t){return t.id}).attr("rx",3).attr("ry",3).attr("x",function(t){return t.milestone?k(t.startTime)+b+.5*(k(t.endTime)-k(t.startTime))-.5*y:k(t.startTime)+b}).attr("y",function(t,f){return f=t.order,t.vert?i.gridLineStartPadding:f*w+x}).attr("width",function(t){return t.milestone?y:t.vert?.08*y:k(t.renderEndTime||t.endTime)-k(t.startTime)}).attr("height",function(t){return t.vert?L.length*(i.barHeight+i.barGap)+2*i.barHeight:y}).attr("transform-origin",function(t,f){return f=t.order,(k(t.startTime)+b+.5*(k(t.endTime)-k(t.startTime))).toString()+"px "+(f*w+x+.5*y).toString()+"px"}).attr("class",function(t){let f="";t.classes.length>0&&(f=t.classes.join(" "));let a=0;for(const[Y,O]of F.entries())t.type===O&&(a=Y%i.numberSectionStyles);let _="";return t.active?t.crit?_+=" activeCrit":_=" active":t.done?_=t.crit?" doneCrit":" done":t.crit&&(_+=" crit"),_.length===0&&(_=" task"),t.milestone&&(_=" milestone "+_),t.vert&&(_=" vert "+_),_+=a,_+=" "+f,"task"+_}),d.append("text").attr("id",function(t){return t.id+"-text"}).text(function(t){return t.task}).attr("font-size",i.fontSize).attr("x",function(t){let f=k(t.startTime),a=k(t.renderEndTime||t.endTime);if(t.milestone&&(f+=.5*(k(t.endTime)-k(t.startTime))-.5*y,a=f+y),t.vert)return k(t.startTime)+b;const _=this.getBBox().width;return _>a-f?a+_+1.5*i.leftPadding>m?f+b-5:a+b+5:(a-f)/2+f+b}).attr("y",function(t,f){return t.vert?i.gridLineStartPadding+L.length*(i.barHeight+i.barGap)+60:t.order*w+i.barHeight/2+(i.fontSize/2-2)+x}).attr("text-height",y).attr("class",function(t){const f=k(t.startTime);let a=k(t.endTime);t.milestone&&(a=f+y);const _=this.getBBox().width;let Y="";t.classes.length>0&&(Y=t.classes.join(" "));let O=0;for(const[q,tt]of F.entries())t.type===tt&&(O=q%i.numberSectionStyles);let C="";return t.active&&(C=t.crit?"activeCritText"+O:"activeText"+O),t.done?C=t.crit?C+" doneCritText"+O:C+" doneText"+O:t.crit&&(C=C+" critText"+O),t.milestone&&(C+=" milestoneText"),t.vert&&(C+=" vertText"),_>a-f?a+_+1.5*i.leftPadding>m?Y+" taskTextOutsideLeft taskTextOutside"+O+" "+C:Y+" taskTextOutsideRight taskTextOutside"+O+" "+C+" width-"+_:Y+" taskText taskText"+O+" "+C+" width-"+_}),ot().securityLevel==="sandbox"){let t;t=bt("#i"+n);const f=t.nodes()[0].contentDocument;d.filter(function(a){return T.has(a.id)}).each(function(a){var _=f.querySelector("#"+a.id),Y=f.querySelector("#"+a.id+"-text");const O=_.parentNode;var C=f.createElement("a");C.setAttribute("xlink:href",T.get(a.id)),C.setAttribute("target","_top"),O.appendChild(C),C.appendChild(_),C.appendChild(Y)})}}function z(p,w,x,b,y,o,m,h){if(m.length===0&&h.length===0)return;let d,T;for(const{startTime:Y,endTime:O}of o)(d===void 0||Y<d)&&(d=Y),(T===void 0||O>T)&&(T=O);if(!d||!T)return;if(V(T).diff(V(d),"year")>5)return void mt.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");const t=r.db.getDateFormat(),f=[];let a=null,_=V(d);for(;_.valueOf()<=T;)r.db.isInvalidDate(_,t,m,h)?a?a.end=_:a={start:_,end:_}:a&&(f.push(a),a=null),_=_.add(1,"d");g.append("g").selectAll("rect").data(f).enter().append("rect").attr("id",Y=>"exclude-"+Y.start.format("YYYY-MM-DD")).attr("x",Y=>k(Y.start.startOf("day"))+x).attr("y",i.gridLineStartPadding).attr("width",Y=>k(Y.end.endOf("day"))-k(Y.start.startOf("day"))).attr("height",y-w-i.gridLineStartPadding).attr("transform-origin",function(Y,O){return(k(Y.start)+x+.5*(k(Y.end)-k(Y.start))).toString()+"px "+(O*p+.5*y).toString()+"px"}).attr("class","exclude-range")}function j(p,w,x,b){const y=r.db.getDateFormat(),o=r.db.getAxisFormat();let m;m=o||(y==="D"?"%d":i.axisFormat??"%Y-%m-%d");let h=(d=k,jt(3,d)).tickSize(-b+w+i.gridLineStartPadding).tickFormat(Wt(m));var d;const T=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(r.db.getTickInterval()||i.tickInterval);if(T!==null){const t=T[1],f=T[2],a=r.db.getWeekday()||i.weekday;switch(f){case"millisecond":h.ticks(Nt.every(t));break;case"second":h.ticks(Ht.every(t));break;case"minute":h.ticks(Bt.every(t));break;case"hour":h.ticks(zt.every(t));break;case"day":h.ticks(Pt.every(t));break;case"week":h.ticks(ce[a].every(t));break;case"month":h.ticks(Ot.every(t))}}if(g.append("g").attr("class","grid").attr("transform","translate("+p+", "+(b-50)+")").call(h).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),r.db.topAxisEnabled()||i.topAxis){let t=function(f){return jt(1,f)}(k).tickSize(-b+w+i.gridLineStartPadding).tickFormat(Wt(m));if(T!==null){const f=T[1],a=T[2],_=r.db.getWeekday()||i.weekday;switch(a){case"millisecond":t.ticks(Nt.every(f));break;case"second":t.ticks(Ht.every(f));break;case"minute":t.ticks(Bt.every(f));break;case"hour":t.ticks(zt.every(f));break;case"day":t.ticks(Pt.every(f));break;case"week":t.ticks(ce[_].every(f));break;case"month":t.ticks(Ot.every(f))}}g.append("g").attr("class","grid").attr("transform","translate("+p+", "+w+")").call(t).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}function B(p,w){let x=0;const b=Object.keys(S).map(y=>[y,S[y]]);g.append("g").selectAll("text").data(b).enter().append(function(y){const o=y[0].split(ke.lineBreakRegex),m=-(o.length-1)/2,h=v.createElementNS("http://www.w3.org/2000/svg","text");h.setAttribute("dy",m+"em");for(const[d,T]of o.entries()){const t=v.createElementNS("http://www.w3.org/2000/svg","tspan");t.setAttribute("alignment-baseline","central"),t.setAttribute("x","10"),d>0&&t.setAttribute("dy","1em"),t.textContent=T,h.appendChild(t)}return h}).attr("x",10).attr("y",function(y,o){if(!(o>0))return y[1]*p/2+w;for(let m=0;m<o;m++)return x+=b[o-1][1],y[1]*p/2+x*p+w}).attr("font-size",i.sectionFontSize).attr("class",function(y){for(const[o,m]of F.entries())if(y[0]===m)return"sectionTitle sectionTitle"+o%i.numberSectionStyles;return"sectionTitle"})}function W(p,w,x,b){const y=r.db.getTodayMarker();if(y==="off")return;const o=g.append("g").attr("class","today"),m=new Date,h=o.append("line");h.attr("x1",k(m)+p).attr("x2",k(m)+p).attr("y1",i.titleTopMargin).attr("y2",b-i.titleTopMargin).attr("class","today"),y!==""&&h.attr("style",y.replace(/,/g,";"))}function I(p){const w={},x=[];for(let b=0,y=p.length;b<y;++b)Object.prototype.hasOwnProperty.call(w,p[b])||(w[p[b]]=!0,x.push(p[b]));return x}l(M,"taskCompare"),L.sort(M),E(L,K,D),ye(g,D,K,i.useMaxWidth),g.append("text").text(r.db.getDiagramTitle()).attr("x",K/2).attr("y",i.titleTopMargin).attr("class","titleText"),l(E,"makeGantt"),l(P,"drawRects"),l(z,"drawExcludeDays"),l(j,"makeGrid"),l(B,"vertLabels"),l(W,"drawToday"),l(I,"checkUnique")},"draw")},styles:l(e=>`
  .mermaid-main-font {
        font-family: ${e.fontFamily};
  }

  .exclude-range {
    fill: ${e.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${e.sectionBkgColor};
  }

  .section2 {
    fill: ${e.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${e.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${e.titleColor};
  }

  .sectionTitle1 {
    fill: ${e.titleColor};
  }

  .sectionTitle2 {
    fill: ${e.titleColor};
  }

  .sectionTitle3 {
    fill: ${e.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${e.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${e.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${e.fontFamily};
    fill: ${e.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${e.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${e.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${e.taskTextDarkColor};
    text-anchor: start;
    font-family: ${e.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${e.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${e.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${e.taskBkgColor};
    stroke: ${e.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${e.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${e.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${e.activeTaskBkgColor};
    stroke: ${e.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${e.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${e.doneTaskBorderColor};
    fill: ${e.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${e.taskTextDarkColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  .vert {
    stroke: ${e.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${e.vertLineColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${e.titleColor||e.textColor};
    font-family: ${e.fontFamily};
  }
`,"getStyles")};export{Mn as diagram};
