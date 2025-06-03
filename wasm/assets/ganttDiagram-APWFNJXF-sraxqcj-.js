import{_ as d,g as le,s as de,q as ue,p as he,a as fe,b as me,c as ot,d as ye,am as V,l as mt,j as ke,i as pe,y as ge,u as be}from"./mermaid-Da56-Xo5.js";import{y as gt,bS as Wt,bK as Ot,bJ as xe,cn as Te,co as ve,cp as we,cq as _e,cr as De,cs as $e,bI as Pt,bH as zt,bG as Bt,bw as Nt,bF as Ht}from"./index-aRpq2G87.js";import{s as bt}from"./transform-Cyp0GDF-.js";import{t as Se,m as Ce,a as Ee,i as Me}from"./time-DYh8gDo3.js";import{l as Ae}from"./linear-QODJVAWE.js";import"./step-BwsUM5iJ.js";import"./timer-DFzT7np-.js";import"./init-DLRA0X12.js";function Ie(t){return t}var jt=1e-6;function Ye(t){return"translate("+t+",0)"}function Le(t){return"translate(0,"+t+")"}function Fe(t){return n=>+t(n)}function We(t,n){return n=Math.max(0,t.bandwidth()-2*n)/2,t.round()&&(n=Math.round(n)),r=>+t(r)+n}function Oe(){return!this.__axis}function Gt(t,n){var r=[],s=null,i=null,c=6,h=6,A=3,x=typeof window<"u"&&window.devicePixelRatio>1?0:.5,$=t===1||t===4?-1:1,Y=t===4||t===2?"x":"y",W=t===1||t===3?Ye:Le;function C(_){var k=s??(n.ticks?n.ticks.apply(n,r):n.domain()),y=i??(n.tickFormat?n.tickFormat.apply(n,r):Ie),E=Math.max(c,0)+A,M=n.range(),P=+M[0]+x,z=+M[M.length-1]+x,G=(n.bandwidth?We:Fe)(n.copy(),x),B=_.selection?_.selection():_,O=B.selectAll(".domain").data([null]),L=B.selectAll(".tick").data(k,n).order(),p=L.exit(),w=L.enter().append("g").attr("class","tick"),g=L.select("line"),b=L.select("text");O=O.merge(O.enter().insert("path",".tick").attr("class","domain").attr("stroke","currentColor")),L=L.merge(w),g=g.merge(w.append("line").attr("stroke","currentColor").attr(Y+"2",$*c)),b=b.merge(w.append("text").attr("fill","currentColor").attr(Y,$*E).attr("dy",t===1?"0em":t===3?"0.71em":"0.32em")),_!==B&&(O=O.transition(_),L=L.transition(_),g=g.transition(_),b=b.transition(_),p=p.transition(_).attr("opacity",jt).attr("transform",function(f){return isFinite(f=G(f))?W(f+x):this.getAttribute("transform")}),w.attr("opacity",jt).attr("transform",function(f){var o=this.parentNode.__axis;return W((o&&isFinite(o=o(f))?o:G(f))+x)})),p.remove(),O.attr("d",t===4||t===2?h?"M"+$*h+","+P+"H"+x+"V"+z+"H"+$*h:"M"+x+","+P+"V"+z:h?"M"+P+","+$*h+"V"+x+"H"+z+"V"+$*h:"M"+P+","+x+"H"+z),L.attr("opacity",1).attr("transform",function(f){return W(G(f)+x)}),g.attr(Y+"2",$*c),b.attr(Y,$*E).text(y),B.filter(Oe).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",t===2?"start":t===4?"end":"middle"),B.each(function(){this.__axis=G})}return C.scale=function(_){return arguments.length?(n=_,C):n},C.ticks=function(){return r=Array.from(arguments),C},C.tickArguments=function(_){return arguments.length?(r=_==null?[]:Array.from(_),C):r.slice()},C.tickValues=function(_){return arguments.length?(s=_==null?null:Array.from(_),C):s&&s.slice()},C.tickFormat=function(_){return arguments.length?(i=_,C):i},C.tickSize=function(_){return arguments.length?(c=h=+_,C):c},C.tickSizeInner=function(_){return arguments.length?(c=+_,C):c},C.tickSizeOuter=function(_){return arguments.length?(h=+_,C):h},C.tickPadding=function(_){return arguments.length?(A=+_,C):A},C.offset=function(_){return arguments.length?(x=+_,C):x},C}var Vt,yt,Rt={exports:{}};const Pe=gt(Vt?Rt.exports:(Vt=1,Rt.exports=(yt="day",function(t,n,r){var s=function(h){return h.add(4-h.isoWeekday(),yt)},i=n.prototype;i.isoWeekYear=function(){return s(this).year()},i.isoWeek=function(h){if(!this.$utils().u(h))return this.add(7*(h-this.isoWeek()),yt);var A,x,$,Y=s(this),W=(A=this.isoWeekYear(),$=4-(x=(this.$u?r.utc:r)().year(A).startOf("year")).isoWeekday(),x.isoWeekday()>4&&($+=7),x.add($,yt));return Y.diff(W,"week")+1},i.isoWeekday=function(h){return this.$utils().u(h)?this.day()||7:this.day(this.day()%7?h:h-7)};var c=i.startOf;i.startOf=function(h,A){var x=this.$utils(),$=!!x.u(A)||A;return x.p(h)==="isoweek"?$?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):c.bind(this)(h,A)}})));var Zt,qt={exports:{}},ze=(Zt||(Zt=1,qt.exports=function(){var t={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},n=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,r=/\d/,s=/\d\d/,i=/\d\d?/,c=/\d*[^-_:/,()\s\d]+/,h={},A=function(k){return(k=+k)+(k>68?1900:2e3)},x=function(k){return function(y){this[k]=+y}},$=[/[+-]\d\d:?(\d\d)?|Z/,function(k){(this.zone||(this.zone={})).offset=function(y){if(!y||y==="Z")return 0;var E=y.match(/([+-]|\d\d)/g),M=60*E[1]+(+E[2]||0);return M===0?0:E[0]==="+"?-M:M}(k)}],Y=function(k){var y=h[k];return y&&(y.indexOf?y:y.s.concat(y.f))},W=function(k,y){var E,M=h.meridiem;if(M){for(var P=1;P<=24;P+=1)if(k.indexOf(M(P,0,y))>-1){E=P>12;break}}else E=k===(y?"pm":"PM");return E},C={A:[c,function(k){this.afternoon=W(k,!1)}],a:[c,function(k){this.afternoon=W(k,!0)}],Q:[r,function(k){this.month=3*(k-1)+1}],S:[r,function(k){this.milliseconds=100*+k}],SS:[s,function(k){this.milliseconds=10*+k}],SSS:[/\d{3}/,function(k){this.milliseconds=+k}],s:[i,x("seconds")],ss:[i,x("seconds")],m:[i,x("minutes")],mm:[i,x("minutes")],H:[i,x("hours")],h:[i,x("hours")],HH:[i,x("hours")],hh:[i,x("hours")],D:[i,x("day")],DD:[s,x("day")],Do:[c,function(k){var y=h.ordinal,E=k.match(/\d+/);if(this.day=E[0],y)for(var M=1;M<=31;M+=1)y(M).replace(/\[|\]/g,"")===k&&(this.day=M)}],w:[i,x("week")],ww:[s,x("week")],M:[i,x("month")],MM:[s,x("month")],MMM:[c,function(k){var y=Y("months"),E=(Y("monthsShort")||y.map(function(M){return M.slice(0,3)})).indexOf(k)+1;if(E<1)throw new Error;this.month=E%12||E}],MMMM:[c,function(k){var y=Y("months").indexOf(k)+1;if(y<1)throw new Error;this.month=y%12||y}],Y:[/[+-]?\d+/,x("year")],YY:[s,function(k){this.year=A(k)}],YYYY:[/\d{4}/,x("year")],Z:$,ZZ:$};function _(k){var y,E;y=k,E=h&&h.formats;for(var M=(k=y.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,function(p,w,g){var b=g&&g.toUpperCase();return w||E[g]||t[g]||E[b].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,function(f,o,m){return o||m.slice(1)})})).match(n),P=M.length,z=0;z<P;z+=1){var G=M[z],B=C[G],O=B&&B[0],L=B&&B[1];M[z]=L?{regex:O,parser:L}:G.replace(/^\[|\]$/g,"")}return function(p){for(var w={},g=0,b=0;g<P;g+=1){var f=M[g];if(typeof f=="string")b+=f.length;else{var o=f.regex,m=f.parser,u=p.slice(b),l=o.exec(u)[0];m.call(w,l),p=p.replace(l,"")}}return function(T){var e=T.afternoon;if(e!==void 0){var v=T.hours;e?v<12&&(T.hours+=12):v===12&&(T.hours=0),delete T.afternoon}}(w),w}}return function(k,y,E){E.p.customParseFormat=!0,k&&k.parseTwoDigitYear&&(A=k.parseTwoDigitYear);var M=y.prototype,P=M.parse;M.parse=function(z){var G=z.date,B=z.utc,O=z.args;this.$u=B;var L=O[1];if(typeof L=="string"){var p=O[2]===!0,w=O[3]===!0,g=p||w,b=O[2];w&&(b=O[2]),h=this.$locale(),!p&&b&&(h=E.Ls[b]),this.$d=function(u,l,T,e){try{if(["x","X"].indexOf(l)>-1)return new Date((l==="X"?1e3:1)*u);var v=_(l)(u),a=v.year,D=v.month,I=v.day,F=v.hours,S=v.minutes,q=v.seconds,tt=v.milliseconds,st=v.zone,ft=v.week,rt=new Date,H=I||(a||D?1:rt.getDate()),U=a||rt.getFullYear(),j=0;a&&!D||(j=D>0?D-1:rt.getMonth());var et,X=F||0,R=S||0,at=q||0,J=tt||0;return st?new Date(Date.UTC(U,j,H,X,R,at,J+60*st.offset*1e3)):T?new Date(Date.UTC(U,j,H,X,R,at,J)):(et=new Date(U,j,H,X,R,at,J),ft&&(et=e(et).week(ft).toDate()),et)}catch{return new Date("")}}(G,L,B,E),this.init(),b&&b!==!0&&(this.$L=this.locale(b).$L),g&&G!=this.format(L)&&(this.$d=new Date("")),h={}}else if(L instanceof Array)for(var f=L.length,o=1;o<=f;o+=1){O[1]=L[o-1];var m=E.apply(this,O);if(m.isValid()){this.$d=m.$d,this.$L=m.$L,this.init();break}o===f&&(this.$d=new Date(""))}else P.call(this,z)}}}()),qt.exports);const Be=gt(ze);var Ut,Qt={exports:{}},Ne=(Ut||(Ut=1,Qt.exports=function(t,n){var r=n.prototype,s=r.format;r.format=function(i){var c=this,h=this.$locale();if(!this.isValid())return s.bind(this)(i);var A=this.$utils(),x=(i||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,function($){switch($){case"Q":return Math.ceil((c.$M+1)/3);case"Do":return h.ordinal(c.$D);case"gggg":return c.weekYear();case"GGGG":return c.isoWeekYear();case"wo":return h.ordinal(c.week(),"W");case"w":case"ww":return A.s(c.week(),$==="w"?1:2,"0");case"W":case"WW":return A.s(c.isoWeek(),$==="W"?1:2,"0");case"k":case"kk":return A.s(String(c.$H===0?24:c.$H),$==="k"?1:2,"0");case"X":return Math.floor(c.$d.getTime()/1e3);case"x":return c.$d.getTime();case"z":return"["+c.offsetName()+"]";case"zzz":return"["+c.offsetName("long")+"]";default:return $}});return s.bind(this)(x)}}),Qt.exports);const He=gt(Ne);var xt=function(){var t=d(function(o,m,u,l){for(u=u||{},l=o.length;l--;u[o[l]]=m);return u},"o"),n=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],r=[1,26],s=[1,27],i=[1,28],c=[1,29],h=[1,30],A=[1,31],x=[1,32],$=[1,33],Y=[1,34],W=[1,9],C=[1,10],_=[1,11],k=[1,12],y=[1,13],E=[1,14],M=[1,15],P=[1,16],z=[1,19],G=[1,20],B=[1,21],O=[1,22],L=[1,23],p=[1,25],w=[1,35],g={trace:d(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:d(function(o,m,u,l,T,e,v){var a=e.length-1;switch(T){case 1:return e[a-1];case 2:case 6:case 7:this.$=[];break;case 3:e[a-1].push(e[a]),this.$=e[a-1];break;case 4:case 5:this.$=e[a];break;case 8:l.setWeekday("monday");break;case 9:l.setWeekday("tuesday");break;case 10:l.setWeekday("wednesday");break;case 11:l.setWeekday("thursday");break;case 12:l.setWeekday("friday");break;case 13:l.setWeekday("saturday");break;case 14:l.setWeekday("sunday");break;case 15:l.setWeekend("friday");break;case 16:l.setWeekend("saturday");break;case 17:l.setDateFormat(e[a].substr(11)),this.$=e[a].substr(11);break;case 18:l.enableInclusiveEndDates(),this.$=e[a].substr(18);break;case 19:l.TopAxis(),this.$=e[a].substr(8);break;case 20:l.setAxisFormat(e[a].substr(11)),this.$=e[a].substr(11);break;case 21:l.setTickInterval(e[a].substr(13)),this.$=e[a].substr(13);break;case 22:l.setExcludes(e[a].substr(9)),this.$=e[a].substr(9);break;case 23:l.setIncludes(e[a].substr(9)),this.$=e[a].substr(9);break;case 24:l.setTodayMarker(e[a].substr(12)),this.$=e[a].substr(12);break;case 27:l.setDiagramTitle(e[a].substr(6)),this.$=e[a].substr(6);break;case 28:this.$=e[a].trim(),l.setAccTitle(this.$);break;case 29:case 30:this.$=e[a].trim(),l.setAccDescription(this.$);break;case 31:l.addSection(e[a].substr(8)),this.$=e[a].substr(8);break;case 33:l.addTask(e[a-1],e[a]),this.$="task";break;case 34:this.$=e[a-1],l.setClickEvent(e[a-1],e[a],null);break;case 35:this.$=e[a-2],l.setClickEvent(e[a-2],e[a-1],e[a]);break;case 36:this.$=e[a-2],l.setClickEvent(e[a-2],e[a-1],null),l.setLink(e[a-2],e[a]);break;case 37:this.$=e[a-3],l.setClickEvent(e[a-3],e[a-2],e[a-1]),l.setLink(e[a-3],e[a]);break;case 38:this.$=e[a-2],l.setClickEvent(e[a-2],e[a],null),l.setLink(e[a-2],e[a-1]);break;case 39:this.$=e[a-3],l.setClickEvent(e[a-3],e[a-1],e[a]),l.setLink(e[a-3],e[a-2]);break;case 40:this.$=e[a-1],l.setLink(e[a-1],e[a]);break;case 41:case 47:this.$=e[a-1]+" "+e[a];break;case 42:case 43:case 45:this.$=e[a-2]+" "+e[a-1]+" "+e[a];break;case 44:case 46:this.$=e[a-3]+" "+e[a-2]+" "+e[a-1]+" "+e[a]}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(n,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:r,13:s,14:i,15:c,16:h,17:A,18:x,19:18,20:$,21:Y,22:W,23:C,24:_,25:k,26:y,27:E,28:M,29:P,30:z,31:G,33:B,35:O,36:L,37:24,38:p,40:w},t(n,[2,7],{1:[2,1]}),t(n,[2,3]),{9:36,11:17,12:r,13:s,14:i,15:c,16:h,17:A,18:x,19:18,20:$,21:Y,22:W,23:C,24:_,25:k,26:y,27:E,28:M,29:P,30:z,31:G,33:B,35:O,36:L,37:24,38:p,40:w},t(n,[2,5]),t(n,[2,6]),t(n,[2,17]),t(n,[2,18]),t(n,[2,19]),t(n,[2,20]),t(n,[2,21]),t(n,[2,22]),t(n,[2,23]),t(n,[2,24]),t(n,[2,25]),t(n,[2,26]),t(n,[2,27]),{32:[1,37]},{34:[1,38]},t(n,[2,30]),t(n,[2,31]),t(n,[2,32]),{39:[1,39]},t(n,[2,8]),t(n,[2,9]),t(n,[2,10]),t(n,[2,11]),t(n,[2,12]),t(n,[2,13]),t(n,[2,14]),t(n,[2,15]),t(n,[2,16]),{41:[1,40],43:[1,41]},t(n,[2,4]),t(n,[2,28]),t(n,[2,29]),t(n,[2,33]),t(n,[2,34],{42:[1,42],43:[1,43]}),t(n,[2,40],{41:[1,44]}),t(n,[2,35],{43:[1,45]}),t(n,[2,36]),t(n,[2,38],{42:[1,46]}),t(n,[2,37]),t(n,[2,39])],defaultActions:{},parseError:d(function(o,m){if(!m.recoverable){var u=new Error(o);throw u.hash=m,u}this.trace(o)},"parseError"),parse:d(function(o){var m=this,u=[0],l=[],T=[null],e=[],v=this.table,a="",D=0,I=0,F=e.slice.call(arguments,1),S=Object.create(this.lexer),q={yy:{}};for(var tt in this.yy)Object.prototype.hasOwnProperty.call(this.yy,tt)&&(q.yy[tt]=this.yy[tt]);S.setInput(o,q.yy),q.yy.lexer=S,q.yy.parser=this,S.yylloc===void 0&&(S.yylloc={});var st=S.yylloc;e.push(st);var ft=S.options&&S.options.ranges;function rt(){var Z;return typeof(Z=l.pop()||S.lex()||1)!="number"&&(Z instanceof Array&&(Z=(l=Z).pop()),Z=m.symbols_[Z]||Z),Z}typeof q.yy.parseError=="function"?this.parseError=q.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError,d(function(Z){u.length=u.length-2*Z,T.length=T.length-Z,e.length=e.length-Z},"popStack"),d(rt,"lex");for(var H,U,j,et,X,R,at,J,nt={};;){if(U=u[u.length-1],this.defaultActions[U]?j=this.defaultActions[U]:(H==null&&(H=rt()),j=v[U]&&v[U][H]),j===void 0||!j.length||!j[0]){var Ft="";for(X in J=[],v[U])this.terminals_[X]&&X>2&&J.push("'"+this.terminals_[X]+"'");Ft=S.showPosition?"Parse error on line "+(D+1)+`:
`+S.showPosition()+`
Expecting `+J.join(", ")+", got '"+(this.terminals_[H]||H)+"'":"Parse error on line "+(D+1)+": Unexpected "+(H==1?"end of input":"'"+(this.terminals_[H]||H)+"'"),this.parseError(Ft,{text:S.match,token:this.terminals_[H]||H,line:S.yylineno,loc:st,expected:J})}if(j[0]instanceof Array&&j.length>1)throw new Error("Parse Error: multiple actions possible at state: "+U+", token: "+H);switch(j[0]){case 1:u.push(H),T.push(S.yytext),e.push(S.yylloc),u.push(j[1]),H=null,I=S.yyleng,a=S.yytext,D=S.yylineno,st=S.yylloc;break;case 2:if(R=this.productions_[j[1]][1],nt.$=T[T.length-R],nt._$={first_line:e[e.length-(R||1)].first_line,last_line:e[e.length-1].last_line,first_column:e[e.length-(R||1)].first_column,last_column:e[e.length-1].last_column},ft&&(nt._$.range=[e[e.length-(R||1)].range[0],e[e.length-1].range[1]]),(et=this.performAction.apply(nt,[a,I,D,q.yy,j[1],T,e].concat(F)))!==void 0)return et;R&&(u=u.slice(0,-1*R*2),T=T.slice(0,-1*R),e=e.slice(0,-1*R)),u.push(this.productions_[j[1]][0]),T.push(nt.$),e.push(nt._$),at=v[u[u.length-2]][u[u.length-1]],u.push(at);break;case 3:return!0}}return!0},"parse")},b=function(){return{EOF:1,parseError:d(function(o,m){if(!this.yy.parser)throw new Error(o);this.yy.parser.parseError(o,m)},"parseError"),setInput:d(function(o,m){return this.yy=m||this.yy||{},this._input=o,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:d(function(){var o=this._input[0];return this.yytext+=o,this.yyleng++,this.offset++,this.match+=o,this.matched+=o,o.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),o},"input"),unput:d(function(o){var m=o.length,u=o.split(/(?:\r\n?|\n)/g);this._input=o+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-m),this.offset-=m;var l=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),u.length-1&&(this.yylineno-=u.length-1);var T=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:u?(u.length===l.length?this.yylloc.first_column:0)+l[l.length-u.length].length-u[0].length:this.yylloc.first_column-m},this.options.ranges&&(this.yylloc.range=[T[0],T[0]+this.yyleng-m]),this.yyleng=this.yytext.length,this},"unput"),more:d(function(){return this._more=!0,this},"more"),reject:d(function(){return this.options.backtrack_lexer?(this._backtrack=!0,this):this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"reject"),less:d(function(o){this.unput(this.match.slice(o))},"less"),pastInput:d(function(){var o=this.matched.substr(0,this.matched.length-this.match.length);return(o.length>20?"...":"")+o.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:d(function(){var o=this.match;return o.length<20&&(o+=this._input.substr(0,20-o.length)),(o.substr(0,20)+(o.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:d(function(){var o=this.pastInput(),m=new Array(o.length+1).join("-");return o+this.upcomingInput()+`
`+m+"^"},"showPosition"),test_match:d(function(o,m){var u,l,T;if(this.options.backtrack_lexer&&(T={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(T.yylloc.range=this.yylloc.range.slice(0))),(l=o[0].match(/(?:\r\n?|\n).*/g))&&(this.yylineno+=l.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:l?l[l.length-1].length-l[l.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+o[0].length},this.yytext+=o[0],this.match+=o[0],this.matches=o,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(o[0].length),this.matched+=o[0],u=this.performAction.call(this,this.yy,this,m,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),u)return u;if(this._backtrack){for(var e in T)this[e]=T[e];return!1}return!1},"test_match"),next:d(function(){if(this.done)return this.EOF;var o,m,u,l;this._input||(this.done=!0),this._more||(this.yytext="",this.match="");for(var T=this._currentRules(),e=0;e<T.length;e++)if((u=this._input.match(this.rules[T[e]]))&&(!m||u[0].length>m[0].length)){if(m=u,l=e,this.options.backtrack_lexer){if((o=this.test_match(u,T[e]))!==!1)return o;if(this._backtrack){m=!1;continue}return!1}if(!this.options.flex)break}return m?(o=this.test_match(m,T[l]))!==!1&&o:this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:d(function(){var o=this.next();return o||this.lex()},"lex"),begin:d(function(o){this.conditionStack.push(o)},"begin"),popState:d(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:d(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:d(function(o){return(o=this.conditionStack.length-1-Math.abs(o||0))>=0?this.conditionStack[o]:"INITIAL"},"topState"),pushState:d(function(o){this.begin(o)},"pushState"),stateStackSize:d(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:d(function(o,m,u,l){switch(u){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:case 15:case 18:case 21:case 24:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:case 9:case 10:case 12:case 13:break;case 11:return 10;case 14:this.begin("href");break;case 16:return 43;case 17:this.begin("callbackname");break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 22:return 42;case 23:this.begin("click");break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}}}();function f(){this.yy={}}return g.lexer=b,d(f,"Parser"),f.prototype=g,g.Parser=f,new f}();xt.parser=xt;var je=xt;V.extend(Pe),V.extend(Be),V.extend(He);var Tt,kt,Xt={friday:5,saturday:6},Q="",vt="",wt=void 0,_t="",dt=[],ut=[],Dt=new Map,$t=[],St=[],ct="",Ct="",Jt=["active","done","crit","milestone"],Et=[],ht=!1,Mt=!1,At="sunday",pt="saturday",It=0,Ge=d(function(){$t=[],St=[],ct="",Et=[],ie=0,Tt=void 0,kt=void 0,N=[],Q="",vt="",Ct="",wt=void 0,_t="",dt=[],ut=[],ht=!1,Mt=!1,It=0,Dt=new Map,ge(),At="sunday",pt="saturday"},"clear"),Ve=d(function(t){vt=t},"setAxisFormat"),Re=d(function(){return vt},"getAxisFormat"),Ze=d(function(t){wt=t},"setTickInterval"),qe=d(function(){return wt},"getTickInterval"),Ue=d(function(t){_t=t},"setTodayMarker"),Qe=d(function(){return _t},"getTodayMarker"),Xe=d(function(t){Q=t},"setDateFormat"),Je=d(function(){ht=!0},"enableInclusiveEndDates"),Ke=d(function(){return ht},"endDatesAreInclusive"),tn=d(function(){Mt=!0},"enableTopAxis"),en=d(function(){return Mt},"topAxisEnabled"),nn=d(function(t){Ct=t},"setDisplayMode"),sn=d(function(){return Ct},"getDisplayMode"),rn=d(function(){return Q},"getDateFormat"),an=d(function(t){dt=t.toLowerCase().split(/[\s,]+/)},"setIncludes"),on=d(function(){return dt},"getIncludes"),cn=d(function(t){ut=t.toLowerCase().split(/[\s,]+/)},"setExcludes"),ln=d(function(){return ut},"getExcludes"),dn=d(function(){return Dt},"getLinks"),un=d(function(t){ct=t,$t.push(t)},"addSection"),hn=d(function(){return $t},"getSections"),fn=d(function(){let t=re(),n=0;for(;!t&&n<10;)t=re(),n++;return St=N},"getTasks"),Kt=d(function(t,n,r,s){return!s.includes(t.format(n.trim()))&&(!(!r.includes("weekends")||t.isoWeekday()!==Xt[pt]&&t.isoWeekday()!==Xt[pt]+1)||!!r.includes(t.format("dddd").toLowerCase())||r.includes(t.format(n.trim())))},"isInvalidDate"),mn=d(function(t){At=t},"setWeekday"),yn=d(function(){return At},"getWeekday"),kn=d(function(t){pt=t},"setWeekend"),te=d(function(t,n,r,s){if(!r.length||t.manualEndTime)return;let i,c;i=t.startTime instanceof Date?V(t.startTime):V(t.startTime,n,!0),i=i.add(1,"d"),c=t.endTime instanceof Date?V(t.endTime):V(t.endTime,n,!0);const[h,A]=pn(i,c,n,r,s);t.endTime=h.toDate(),t.renderEndTime=A},"checkTaskDates"),pn=d(function(t,n,r,s,i){let c=!1,h=null;for(;t<=n;)c||(h=n.toDate()),c=Kt(t,r,s,i),c&&(n=n.add(1,"d")),t=t.add(1,"d");return[n,h]},"fixTaskDates"),Yt=d(function(t,n,r){r=r.trim();const s=/^after\s+(?<ids>[\d\w- ]+)/.exec(r);if(s!==null){let c=null;for(const A of s.groups.ids.split(" ")){let x=it(A);x!==void 0&&(!c||x.endTime>c.endTime)&&(c=x)}if(c)return c.endTime;const h=new Date;return h.setHours(0,0,0,0),h}let i=V(r,n.trim(),!0);if(i.isValid())return i.toDate();{mt.debug("Invalid date:"+r),mt.debug("With date format:"+n.trim());const c=new Date(r);if(c===void 0||isNaN(c.getTime())||c.getFullYear()<-1e4||c.getFullYear()>1e4)throw new Error("Invalid date:"+r);return c}},"getStartDate"),ee=d(function(t){const n=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return n!==null?[Number.parseFloat(n[1]),n[2]]:[NaN,"ms"]},"parseDuration"),ne=d(function(t,n,r,s=!1){r=r.trim();const i=/^until\s+(?<ids>[\d\w- ]+)/.exec(r);if(i!==null){let $=null;for(const W of i.groups.ids.split(" ")){let C=it(W);C!==void 0&&(!$||C.startTime<$.startTime)&&($=C)}if($)return $.startTime;const Y=new Date;return Y.setHours(0,0,0,0),Y}let c=V(r,n.trim(),!0);if(c.isValid())return s&&(c=c.add(1,"d")),c.toDate();let h=V(t);const[A,x]=ee(r);if(!Number.isNaN(A)){const $=h.add(A,x);$.isValid()&&(h=$)}return h.toDate()},"getEndDate"),ie=0,lt=d(function(t){return t===void 0?"task"+(ie+=1):t},"parseId"),gn=d(function(t,n){let r;r=n.substr(0,1)===":"?n.substr(1,n.length):n;const s=r.split(","),i={};Lt(s,i,Jt);for(let h=0;h<s.length;h++)s[h]=s[h].trim();let c="";switch(s.length){case 1:i.id=lt(),i.startTime=t.endTime,c=s[0];break;case 2:i.id=lt(),i.startTime=Yt(void 0,Q,s[0]),c=s[1];break;case 3:i.id=lt(s[0]),i.startTime=Yt(void 0,Q,s[1]),c=s[2]}return c&&(i.endTime=ne(i.startTime,Q,c,ht),i.manualEndTime=V(c,"YYYY-MM-DD",!0).isValid(),te(i,Q,ut,dt)),i},"compileData"),bn=d(function(t,n){let r;r=n.substr(0,1)===":"?n.substr(1,n.length):n;const s=r.split(","),i={};Lt(s,i,Jt);for(let c=0;c<s.length;c++)s[c]=s[c].trim();switch(s.length){case 1:i.id=lt(),i.startTime={type:"prevTaskEnd",id:t},i.endTime={data:s[0]};break;case 2:i.id=lt(),i.startTime={type:"getStartDate",startData:s[0]},i.endTime={data:s[1]};break;case 3:i.id=lt(s[0]),i.startTime={type:"getStartDate",startData:s[1]},i.endTime={data:s[2]}}return i},"parseData"),N=[],se={},xn=d(function(t,n){const r={section:ct,type:ct,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:n},task:t,classes:[]},s=bn(kt,n);r.raw.startTime=s.startTime,r.raw.endTime=s.endTime,r.id=s.id,r.prevTaskId=kt,r.active=s.active,r.done=s.done,r.crit=s.crit,r.milestone=s.milestone,r.order=It,It++;const i=N.push(r);kt=r.id,se[r.id]=i-1},"addTask"),it=d(function(t){const n=se[t];return N[n]},"findTaskById"),Tn=d(function(t,n){const r={section:ct,type:ct,description:t,task:t,classes:[]},s=gn(Tt,n);r.startTime=s.startTime,r.endTime=s.endTime,r.id=s.id,r.active=s.active,r.done=s.done,r.crit=s.crit,r.milestone=s.milestone,Tt=r,St.push(r)},"addTaskOrg"),re=d(function(){const t=d(function(r){const s=N[r];let i="";switch(N[r].raw.startTime.type){case"prevTaskEnd":{const c=it(s.prevTaskId);s.startTime=c.endTime;break}case"getStartDate":i=Yt(void 0,Q,N[r].raw.startTime.startData),i&&(N[r].startTime=i)}return N[r].startTime&&(N[r].endTime=ne(N[r].startTime,Q,N[r].raw.endTime.data,ht),N[r].endTime&&(N[r].processed=!0,N[r].manualEndTime=V(N[r].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),te(N[r],Q,ut,dt))),N[r].processed},"compileTask");let n=!0;for(const[r,s]of N.entries())t(r),n=n&&s.processed;return n},"compileTasks"),vn=d(function(t,n){let r=n;ot().securityLevel!=="loose"&&(r=pe.sanitizeUrl(n)),t.split(",").forEach(function(s){it(s)!==void 0&&(oe(s,()=>{window.open(r,"_self")}),Dt.set(s,r))}),ae(t,"clickable")},"setLink"),ae=d(function(t,n){t.split(",").forEach(function(r){let s=it(r);s!==void 0&&s.classes.push(n)})},"setClass"),wn=d(function(t,n,r){if(ot().securityLevel!=="loose"||n===void 0)return;let s=[];if(typeof r=="string"){s=r.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let i=0;i<s.length;i++){let c=s[i].trim();c.startsWith('"')&&c.endsWith('"')&&(c=c.substr(1,c.length-2)),s[i]=c}}s.length===0&&s.push(t),it(t)!==void 0&&oe(t,()=>{be.runFunc(n,...s)})},"setClickFun"),oe=d(function(t,n){Et.push(function(){const r=document.querySelector(`[id="${t}"]`);r!==null&&r.addEventListener("click",function(){n()})},function(){const r=document.querySelector(`[id="${t}-text"]`);r!==null&&r.addEventListener("click",function(){n()})})},"pushFun"),_n=d(function(t,n,r){t.split(",").forEach(function(s){wn(s,n,r)}),ae(t,"clickable")},"setClickEvent"),Dn=d(function(t){Et.forEach(function(n){n(t)})},"bindFunctions"),$n={getConfig:d(()=>ot().gantt,"getConfig"),clear:Ge,setDateFormat:Xe,getDateFormat:rn,enableInclusiveEndDates:Je,endDatesAreInclusive:Ke,enableTopAxis:tn,topAxisEnabled:en,setAxisFormat:Ve,getAxisFormat:Re,setTickInterval:Ze,getTickInterval:qe,setTodayMarker:Ue,getTodayMarker:Qe,setAccTitle:me,getAccTitle:fe,setDiagramTitle:he,getDiagramTitle:ue,setDisplayMode:nn,getDisplayMode:sn,setAccDescription:de,getAccDescription:le,addSection:un,getSections:hn,getTasks:fn,addTask:xn,findTaskById:it,addTaskOrg:Tn,setIncludes:an,getIncludes:on,setExcludes:cn,getExcludes:ln,setClickEvent:_n,setLink:vn,getLinks:dn,bindFunctions:Dn,parseDuration:ee,isInvalidDate:Kt,setWeekday:mn,getWeekday:yn,setWeekend:kn};function Lt(t,n,r){let s=!0;for(;s;)s=!1,r.forEach(function(i){const c=new RegExp("^\\s*"+i+"\\s*$");t[0].match(c)&&(n[i]=!0,t.shift(1),s=!0)})}d(Lt,"getTaskTags");var K,Sn=d(function(){mt.debug("Something is calling, setConf, remove the call")},"setConf"),ce={monday:$e,tuesday:De,wednesday:_e,thursday:we,friday:ve,saturday:Te,sunday:xe},Cn=d((t,n)=>{let r=[...t].map(()=>-1/0),s=[...t].sort((c,h)=>c.startTime-h.startTime||c.order-h.order),i=0;for(const c of s)for(let h=0;h<r.length;h++)if(c.startTime>=r[h]){r[h]=c.endTime,c.order=h+n,h>i&&(i=h);break}return i},"getMaxIntersections"),En={parser:je,db:$n,renderer:{setConf:Sn,draw:d(function(t,n,r,s){const i=ot().gantt,c=ot().securityLevel;let h;c==="sandbox"&&(h=bt("#i"+n));const A=bt(c==="sandbox"?h.nodes()[0].contentDocument.body:"body"),x=c==="sandbox"?h.nodes()[0].contentDocument:document,$=x.getElementById(n);(K=$.parentElement.offsetWidth)===void 0&&(K=1200),i.useWidth!==void 0&&(K=i.useWidth);const Y=s.db.getTasks();let W=[];for(const p of Y)W.push(p.type);W=L(W);const C={};let _=2*i.topPadding;if(s.db.getDisplayMode()==="compact"||i.displayMode==="compact"){const p={};for(const g of Y)p[g.section]===void 0?p[g.section]=[g]:p[g.section].push(g);let w=0;for(const g of Object.keys(p)){const b=Cn(p[g],w)+1;w+=b,_+=b*(i.barHeight+i.barGap),C[g]=b}}else{_+=Y.length*(i.barHeight+i.barGap);for(const p of W)C[p]=Y.filter(w=>w.type===p).length}$.setAttribute("viewBox","0 0 "+K+" "+_);const k=A.select(`[id="${n}"]`),y=Se().domain([Ce(Y,function(p){return p.startTime}),Ee(Y,function(p){return p.endTime})]).rangeRound([0,K-i.leftPadding-i.rightPadding]);function E(p,w){const g=p.startTime,b=w.startTime;let f=0;return g>b?f=1:g<b&&(f=-1),f}function M(p,w,g){const b=i.barHeight,f=b+i.barGap,o=i.topPadding,m=i.leftPadding,u=Ae().domain([0,W.length]).range(["#00B9FA","#F95002"]).interpolate(Me);z(f,o,m,w,g,p,s.db.getExcludes(),s.db.getIncludes()),G(m,o,w,g),P(p,f,o,m,b,u,w),B(f,o),O(m,o,w,g)}function P(p,w,g,b,f,o,m){const u=[...new Set(p.map(e=>e.order))].map(e=>p.find(v=>v.order===e));k.append("g").selectAll("rect").data(u).enter().append("rect").attr("x",0).attr("y",function(e,v){return e.order*w+g-2}).attr("width",function(){return m-i.rightPadding/2}).attr("height",w).attr("class",function(e){for(const[v,a]of W.entries())if(e.type===a)return"section section"+v%i.numberSectionStyles;return"section section0"});const l=k.append("g").selectAll("rect").data(p).enter(),T=s.db.getLinks();if(l.append("rect").attr("id",function(e){return e.id}).attr("rx",3).attr("ry",3).attr("x",function(e){return e.milestone?y(e.startTime)+b+.5*(y(e.endTime)-y(e.startTime))-.5*f:y(e.startTime)+b}).attr("y",function(e,v){return e.order*w+g}).attr("width",function(e){return e.milestone?f:y(e.renderEndTime||e.endTime)-y(e.startTime)}).attr("height",f).attr("transform-origin",function(e,v){return v=e.order,(y(e.startTime)+b+.5*(y(e.endTime)-y(e.startTime))).toString()+"px "+(v*w+g+.5*f).toString()+"px"}).attr("class",function(e){let v="";e.classes.length>0&&(v=e.classes.join(" "));let a=0;for(const[I,F]of W.entries())e.type===F&&(a=I%i.numberSectionStyles);let D="";return e.active?e.crit?D+=" activeCrit":D=" active":e.done?D=e.crit?" doneCrit":" done":e.crit&&(D+=" crit"),D.length===0&&(D=" task"),e.milestone&&(D=" milestone "+D),D+=a,D+=" "+v,"task"+D}),l.append("text").attr("id",function(e){return e.id+"-text"}).text(function(e){return e.task}).attr("font-size",i.fontSize).attr("x",function(e){let v=y(e.startTime),a=y(e.renderEndTime||e.endTime);e.milestone&&(v+=.5*(y(e.endTime)-y(e.startTime))-.5*f),e.milestone&&(a=v+f);const D=this.getBBox().width;return D>a-v?a+D+1.5*i.leftPadding>m?v+b-5:a+b+5:(a-v)/2+v+b}).attr("y",function(e,v){return e.order*w+i.barHeight/2+(i.fontSize/2-2)+g}).attr("text-height",f).attr("class",function(e){const v=y(e.startTime);let a=y(e.endTime);e.milestone&&(a=v+f);const D=this.getBBox().width;let I="";e.classes.length>0&&(I=e.classes.join(" "));let F=0;for(const[q,tt]of W.entries())e.type===tt&&(F=q%i.numberSectionStyles);let S="";return e.active&&(S=e.crit?"activeCritText"+F:"activeText"+F),e.done?S=e.crit?S+" doneCritText"+F:S+" doneText"+F:e.crit&&(S=S+" critText"+F),e.milestone&&(S+=" milestoneText"),D>a-v?a+D+1.5*i.leftPadding>m?I+" taskTextOutsideLeft taskTextOutside"+F+" "+S:I+" taskTextOutsideRight taskTextOutside"+F+" "+S+" width-"+D:I+" taskText taskText"+F+" "+S+" width-"+D}),ot().securityLevel==="sandbox"){let e;e=bt("#i"+n);const v=e.nodes()[0].contentDocument;l.filter(function(a){return T.has(a.id)}).each(function(a){var D=v.querySelector("#"+a.id),I=v.querySelector("#"+a.id+"-text");const F=D.parentNode;var S=v.createElement("a");S.setAttribute("xlink:href",T.get(a.id)),S.setAttribute("target","_top"),F.appendChild(S),S.appendChild(D),S.appendChild(I)})}}function z(p,w,g,b,f,o,m,u){if(m.length===0&&u.length===0)return;let l,T;for(const{startTime:I,endTime:F}of o)(l===void 0||I<l)&&(l=I),(T===void 0||F>T)&&(T=F);if(!l||!T)return;if(V(T).diff(V(l),"year")>5)return void mt.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");const e=s.db.getDateFormat(),v=[];let a=null,D=V(l);for(;D.valueOf()<=T;)s.db.isInvalidDate(D,e,m,u)?a?a.end=D:a={start:D,end:D}:a&&(v.push(a),a=null),D=D.add(1,"d");k.append("g").selectAll("rect").data(v).enter().append("rect").attr("id",function(I){return"exclude-"+I.start.format("YYYY-MM-DD")}).attr("x",function(I){return y(I.start)+g}).attr("y",i.gridLineStartPadding).attr("width",function(I){const F=I.end.add(1,"day");return y(F)-y(I.start)}).attr("height",f-w-i.gridLineStartPadding).attr("transform-origin",function(I,F){return(y(I.start)+g+.5*(y(I.end)-y(I.start))).toString()+"px "+(F*p+.5*f).toString()+"px"}).attr("class","exclude-range")}function G(p,w,g,b){let f=(o=y,Gt(3,o)).tickSize(-b+w+i.gridLineStartPadding).tickFormat(Wt(s.db.getAxisFormat()||i.axisFormat||"%Y-%m-%d"));var o;const m=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(s.db.getTickInterval()||i.tickInterval);if(m!==null){const u=m[1],l=m[2],T=s.db.getWeekday()||i.weekday;switch(l){case"millisecond":f.ticks(Ht.every(u));break;case"second":f.ticks(Nt.every(u));break;case"minute":f.ticks(Bt.every(u));break;case"hour":f.ticks(zt.every(u));break;case"day":f.ticks(Pt.every(u));break;case"week":f.ticks(ce[T].every(u));break;case"month":f.ticks(Ot.every(u))}}if(k.append("g").attr("class","grid").attr("transform","translate("+p+", "+(b-50)+")").call(f).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),s.db.topAxisEnabled()||i.topAxis){let u=function(l){return Gt(1,l)}(y).tickSize(-b+w+i.gridLineStartPadding).tickFormat(Wt(s.db.getAxisFormat()||i.axisFormat||"%Y-%m-%d"));if(m!==null){const l=m[1],T=m[2],e=s.db.getWeekday()||i.weekday;switch(T){case"millisecond":u.ticks(Ht.every(l));break;case"second":u.ticks(Nt.every(l));break;case"minute":u.ticks(Bt.every(l));break;case"hour":u.ticks(zt.every(l));break;case"day":u.ticks(Pt.every(l));break;case"week":u.ticks(ce[e].every(l));break;case"month":u.ticks(Ot.every(l))}}k.append("g").attr("class","grid").attr("transform","translate("+p+", "+w+")").call(u).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}function B(p,w){let g=0;const b=Object.keys(C).map(f=>[f,C[f]]);k.append("g").selectAll("text").data(b).enter().append(function(f){const o=f[0].split(ke.lineBreakRegex),m=-(o.length-1)/2,u=x.createElementNS("http://www.w3.org/2000/svg","text");u.setAttribute("dy",m+"em");for(const[l,T]of o.entries()){const e=x.createElementNS("http://www.w3.org/2000/svg","tspan");e.setAttribute("alignment-baseline","central"),e.setAttribute("x","10"),l>0&&e.setAttribute("dy","1em"),e.textContent=T,u.appendChild(e)}return u}).attr("x",10).attr("y",function(f,o){if(!(o>0))return f[1]*p/2+w;for(let m=0;m<o;m++)return g+=b[o-1][1],f[1]*p/2+g*p+w}).attr("font-size",i.sectionFontSize).attr("class",function(f){for(const[o,m]of W.entries())if(f[0]===m)return"sectionTitle sectionTitle"+o%i.numberSectionStyles;return"sectionTitle"})}function O(p,w,g,b){const f=s.db.getTodayMarker();if(f==="off")return;const o=k.append("g").attr("class","today"),m=new Date,u=o.append("line");u.attr("x1",y(m)+p).attr("x2",y(m)+p).attr("y1",i.titleTopMargin).attr("y2",b-i.titleTopMargin).attr("class","today"),f!==""&&u.attr("style",f.replace(/,/g,";"))}function L(p){const w={},g=[];for(let b=0,f=p.length;b<f;++b)Object.prototype.hasOwnProperty.call(w,p[b])||(w[p[b]]=!0,g.push(p[b]));return g}d(E,"taskCompare"),Y.sort(E),M(Y,K,_),ye(k,_,K,i.useMaxWidth),k.append("text").text(s.db.getDiagramTitle()).attr("x",K/2).attr("y",i.titleTopMargin).attr("class","titleText"),d(M,"makeGantt"),d(P,"drawRects"),d(z,"drawExcludeDays"),d(G,"makeGrid"),d(B,"vertLabels"),d(O,"drawToday"),d(L,"checkUnique")},"draw")},styles:d(t=>`
  .mermaid-main-font {
        font-family: ${t.fontFamily};
  }

  .exclude-range {
    fill: ${t.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${t.sectionBkgColor};
  }

  .section2 {
    fill: ${t.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${t.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${t.titleColor};
  }

  .sectionTitle1 {
    fill: ${t.titleColor};
  }

  .sectionTitle2 {
    fill: ${t.titleColor};
  }

  .sectionTitle3 {
    fill: ${t.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${t.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${t.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${t.fontFamily};
    fill: ${t.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${t.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${t.taskTextDarkColor};
    text-anchor: start;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${t.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${t.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${t.taskBkgColor};
    stroke: ${t.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${t.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${t.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${t.activeTaskBkgColor};
    stroke: ${t.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${t.doneTaskBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.doneTaskBkgColor};
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
    fill: ${t.taskTextDarkColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${t.titleColor||t.textColor};
    font-family: ${t.fontFamily};
  }
`,"getStyles")};export{En as diagram};
