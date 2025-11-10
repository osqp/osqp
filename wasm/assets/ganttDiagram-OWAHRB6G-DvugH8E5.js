import{s as at,t as _t}from"./chunk-DZLz74EQ.js";import"./_Uint8Array-D5Z9rM2X.js";import"./isArrayLikeObject-DKyJYtr8.js";import"./memoize-DhF8x4Aa.js";import"./merge-Dvc5opZF.js";import"./preload-helper-BnutJmlU.js";import"./precisionRound-Qxa8hC2h.js";import{t as fe}from"./linear-CPlUB2Oz.js";import{n as me,o as ye,s as ke}from"./time-4HenLWop.js";import"./defaultLocale-YmL2k7Vp.js";import{C as Rt,N as Ut,T as Zt,c as qt,d as pe,f as ge,g as be,h as ve,m as Te,p as xe,t as Qt,u as we,v as Xt,x as Jt}from"./defaultLocale-zQltfbSS.js";import"./purify.es-CPMOFAIu.js";import{o as _e}from"./timer-CTpu0Fa6.js";import{u as Dt}from"./src-BwH0YUPh.js";import"./math-B-uM5inP.js";import"./step-BDtirdXu.js";import{g as De}from"./chunk-S3R3BYOJ-BW8g0Zog.js";import"./init-D34LU3qG.js";import{a as Kt,n as u,r as gt}from"./src-BuVGraFB.js";import{B as $e,C as Se,U as Ce,_ as Me,a as Ee,b as ot,c as Ae,s as Le,v as Ye,z as Ie}from"./chunk-ABZYJK2D-Cm5Mq_1V.js";import{t as Fe}from"./dist-DiomJpDg.js";function We(t){return t}var bt=1,$t=2,St=3,vt=4,te=1e-6;function Oe(t){return"translate("+t+",0)"}function Pe(t){return"translate(0,"+t+")"}function ze(t){return e=>+t(e)}function Be(t,e){return e=Math.max(0,t.bandwidth()-e*2)/2,t.round()&&(e=Math.round(e)),r=>+t(r)+e}function Ne(){return!this.__axis}function ee(t,e){var r=[],n=null,o=null,d=6,m=6,$=3,S=typeof window<"u"&&window.devicePixelRatio>1?0:.5,D=t===bt||t===vt?-1:1,x=t===vt||t===$t?"x":"y",L=t===bt||t===St?Oe:Pe;function _(w){var H=n??(e.ticks?e.ticks.apply(e,r):e.domain()),E=o??(e.tickFormat?e.tickFormat.apply(e,r):We),b=Math.max(d,0)+$,M=e.range(),A=+M[0]+S,I=+M[M.length-1]+S,B=(e.bandwidth?Be:ze)(e.copy(),S),z=w.selection?w.selection():w,j=z.selectAll(".domain").data([null]),W=z.selectAll(".tick").data(H,e).order(),y=W.exit(),v=W.enter().append("g").attr("class","tick"),g=W.select("line"),k=W.select("text");j=j.merge(j.enter().insert("path",".tick").attr("class","domain").attr("stroke","currentColor")),W=W.merge(v),g=g.merge(v.append("line").attr("stroke","currentColor").attr(x+"2",D*d)),k=k.merge(v.append("text").attr("fill","currentColor").attr(x,D*b).attr("dy",t===bt?"0em":t===St?"0.71em":"0.32em")),w!==z&&(j=j.transition(w),W=W.transition(w),g=g.transition(w),k=k.transition(w),y=y.transition(w).attr("opacity",te).attr("transform",function(s){return isFinite(s=B(s))?L(s+S):this.getAttribute("transform")}),v.attr("opacity",te).attr("transform",function(s){var f=this.parentNode.__axis;return L((f&&isFinite(f=f(s))?f:B(s))+S)})),y.remove(),j.attr("d",t===vt||t===$t?m?"M"+D*m+","+A+"H"+S+"V"+I+"H"+D*m:"M"+S+","+A+"V"+I:m?"M"+A+","+D*m+"V"+S+"H"+I+"V"+D*m:"M"+A+","+S+"H"+I),W.attr("opacity",1).attr("transform",function(s){return L(B(s)+S)}),g.attr(x+"2",D*d),k.attr(x,D*b).text(E),z.filter(Ne).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",t===$t?"start":t===vt?"end":"middle"),z.each(function(){this.__axis=B})}return _.scale=function(w){return arguments.length?(e=w,_):e},_.ticks=function(){return r=Array.from(arguments),_},_.tickArguments=function(w){return arguments.length?(r=w==null?[]:Array.from(w),_):r.slice()},_.tickValues=function(w){return arguments.length?(n=w==null?null:Array.from(w),_):n&&n.slice()},_.tickFormat=function(w){return arguments.length?(o=w,_):o},_.tickSize=function(w){return arguments.length?(d=m=+w,_):d},_.tickSizeInner=function(w){return arguments.length?(d=+w,_):d},_.tickSizeOuter=function(w){return arguments.length?(m=+w,_):m},_.tickPadding=function(w){return arguments.length?($=+w,_):$},_.offset=function(w){return arguments.length?(S=+w,_):S},_}function He(t){return ee(bt,t)}function je(t){return ee(St,t)}var Ge=_t(((t,e)=>{(function(r,n){typeof t=="object"&&e!==void 0?e.exports=n():typeof define=="function"&&define.amd?define(n):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_isoWeek=n()})(t,(function(){var r="day";return function(n,o,d){var m=function(D){return D.add(4-D.isoWeekday(),r)},$=o.prototype;$.isoWeekYear=function(){return m(this).year()},$.isoWeek=function(D){if(!this.$utils().u(D))return this.add(7*(D-this.isoWeek()),r);var x,L,_,w,H=m(this),E=(x=this.isoWeekYear(),L=this.$u,_=(L?d.utc:d)().year(x).startOf("year"),w=4-_.isoWeekday(),_.isoWeekday()>4&&(w+=7),_.add(w,r));return H.diff(E,"week")+1},$.isoWeekday=function(D){return this.$utils().u(D)?this.day()||7:this.day(this.day()%7?D:D-7)};var S=$.startOf;$.startOf=function(D,x){var L=this.$utils(),_=!!L.u(x)||x;return L.p(D)==="isoweek"?_?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):S.bind(this)(D,x)}}}))})),Ve=_t(((t,e)=>{(function(r,n){typeof t=="object"&&e!==void 0?e.exports=n():typeof define=="function"&&define.amd?define(n):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_customParseFormat=n()})(t,(function(){var r={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},n=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,o=/\d/,d=/\d\d/,m=/\d\d?/,$=/\d*[^-_:/,()\s\d]+/,S={},D=function(b){return(b=+b)+(b>68?1900:2e3)},x=function(b){return function(M){this[b]=+M}},L=[/[+-]\d\d:?(\d\d)?|Z/,function(b){(this.zone||(this.zone={})).offset=(function(M){if(!M||M==="Z")return 0;var A=M.match(/([+-]|\d\d)/g),I=60*A[1]+(+A[2]||0);return I===0?0:A[0]==="+"?-I:I})(b)}],_=function(b){var M=S[b];return M&&(M.indexOf?M:M.s.concat(M.f))},w=function(b,M){var A,I=S.meridiem;if(I){for(var B=1;B<=24;B+=1)if(b.indexOf(I(B,0,M))>-1){A=B>12;break}}else A=b===(M?"pm":"PM");return A},H={A:[$,function(b){this.afternoon=w(b,!1)}],a:[$,function(b){this.afternoon=w(b,!0)}],Q:[o,function(b){this.month=3*(b-1)+1}],S:[o,function(b){this.milliseconds=100*b}],SS:[d,function(b){this.milliseconds=10*b}],SSS:[/\d{3}/,function(b){this.milliseconds=+b}],s:[m,x("seconds")],ss:[m,x("seconds")],m:[m,x("minutes")],mm:[m,x("minutes")],H:[m,x("hours")],h:[m,x("hours")],HH:[m,x("hours")],hh:[m,x("hours")],D:[m,x("day")],DD:[d,x("day")],Do:[$,function(b){var M=S.ordinal;if(this.day=b.match(/\d+/)[0],M)for(var A=1;A<=31;A+=1)M(A).replace(/\[|\]/g,"")===b&&(this.day=A)}],w:[m,x("week")],ww:[d,x("week")],M:[m,x("month")],MM:[d,x("month")],MMM:[$,function(b){var M=_("months"),A=(_("monthsShort")||M.map((function(I){return I.slice(0,3)}))).indexOf(b)+1;if(A<1)throw Error();this.month=A%12||A}],MMMM:[$,function(b){var M=_("months").indexOf(b)+1;if(M<1)throw Error();this.month=M%12||M}],Y:[/[+-]?\d+/,x("year")],YY:[d,function(b){this.year=D(b)}],YYYY:[/\d{4}/,x("year")],Z:L,ZZ:L};function E(b){for(var M=b,A=S&&S.formats,I=(b=M.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,(function(g,k,s){var f=s&&s.toUpperCase();return k||A[s]||r[s]||A[f].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,(function(h,l,p){return l||p.slice(1)}))}))).match(n),B=I.length,z=0;z<B;z+=1){var j=I[z],W=H[j],y=W&&W[0],v=W&&W[1];I[z]=v?{regex:y,parser:v}:j.replace(/^\[|\]$/g,"")}return function(g){for(var k={},s=0,f=0;s<B;s+=1){var h=I[s];if(typeof h=="string")f+=h.length;else{var l=h.regex,p=h.parser,i=g.slice(f),c=l.exec(i)[0];p.call(k,c),g=g.replace(c,"")}}return(function(a){var C=a.afternoon;if(C!==void 0){var T=a.hours;C?T<12&&(a.hours+=12):T===12&&(a.hours=0),delete a.afternoon}})(k),k}}return function(b,M,A){A.p.customParseFormat=!0,b&&b.parseTwoDigitYear&&(D=b.parseTwoDigitYear);var I=M.prototype,B=I.parse;I.parse=function(z){var j=z.date,W=z.utc,y=z.args;this.$u=W;var v=y[1];if(typeof v=="string"){var g=y[2]===!0,k=y[3]===!0,s=g||k,f=y[2];k&&(f=y[2]),S=this.$locale(),!g&&f&&(S=A.Ls[f]),this.$d=(function(i,c,a,C){try{if(["x","X"].indexOf(c)>-1)return new Date((c==="X"?1e3:1)*i);var T=E(c)(i),Y=T.year,F=T.month,O=T.day,ut=T.hours,P=T.minutes,X=T.seconds,dt=T.milliseconds,nt=T.zone,kt=T.week,ht=new Date,st=O||(Y||F?1:ht.getDate()),G=Y||ht.getFullYear(),tt=0;Y&&!F||(tt=F>0?F-1:ht.getMonth());var R,V=ut||0,it=P||0,q=X||0,et=dt||0;return nt?new Date(Date.UTC(G,tt,st,V,it,q,et+60*nt.offset*1e3)):a?new Date(Date.UTC(G,tt,st,V,it,q,et)):(R=new Date(G,tt,st,V,it,q,et),kt&&(R=C(R).week(kt).toDate()),R)}catch{return new Date("")}})(j,v,W,A),this.init(),f&&f!==!0&&(this.$L=this.locale(f).$L),s&&j!=this.format(v)&&(this.$d=new Date("")),S={}}else if(v instanceof Array)for(var h=v.length,l=1;l<=h;l+=1){y[1]=v[l-1];var p=A.apply(this,y);if(p.isValid()){this.$d=p.$d,this.$L=p.$L,this.init();break}l===h&&(this.$d=new Date(""))}else B.call(this,z)}}}))})),Re=_t(((t,e)=>{(function(r,n){typeof t=="object"&&e!==void 0?e.exports=n():typeof define=="function"&&define.amd?define(n):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_advancedFormat=n()})(t,(function(){return function(r,n){var o=n.prototype,d=o.format;o.format=function(m){var $=this,S=this.$locale();if(!this.isValid())return d.bind(this)(m);var D=this.$utils(),x=(m||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,(function(L){switch(L){case"Q":return Math.ceil(($.$M+1)/3);case"Do":return S.ordinal($.$D);case"gggg":return $.weekYear();case"GGGG":return $.isoWeekYear();case"wo":return S.ordinal($.week(),"W");case"w":case"ww":return D.s($.week(),L==="w"?1:2,"0");case"W":case"WW":return D.s($.isoWeek(),L==="W"?1:2,"0");case"k":case"kk":return D.s(String($.$H===0?24:$.$H),L==="k"?1:2,"0");case"X":return Math.floor($.$d.getTime()/1e3);case"x":return $.$d.getTime();case"z":return"["+$.offsetName()+"]";case"zzz":return"["+$.offsetName("long")+"]";default:return L}}));return d.bind(this)(x)}}}))})),Ue=at(Fe(),1),Z=at(Kt(),1),Ze=at(Ge(),1),qe=at(Ve(),1),Qe=at(Re(),1),Ct=at(Kt(),1),Mt=(function(){var t=u(function(s,f,h,l){for(h||(h={}),l=s.length;l--;h[s[l]]=f);return h},"o"),e=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],r=[1,26],n=[1,27],o=[1,28],d=[1,29],m=[1,30],$=[1,31],S=[1,32],D=[1,33],x=[1,34],L=[1,9],_=[1,10],w=[1,11],H=[1,12],E=[1,13],b=[1,14],M=[1,15],A=[1,16],I=[1,19],B=[1,20],z=[1,21],j=[1,22],W=[1,23],y=[1,25],v=[1,35],g={trace:u(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:u(function(s,f,h,l,p,i,c){var a=i.length-1;switch(p){case 1:return i[a-1];case 2:this.$=[];break;case 3:i[a-1].push(i[a]),this.$=i[a-1];break;case 4:case 5:this.$=i[a];break;case 6:case 7:this.$=[];break;case 8:l.setWeekday("monday");break;case 9:l.setWeekday("tuesday");break;case 10:l.setWeekday("wednesday");break;case 11:l.setWeekday("thursday");break;case 12:l.setWeekday("friday");break;case 13:l.setWeekday("saturday");break;case 14:l.setWeekday("sunday");break;case 15:l.setWeekend("friday");break;case 16:l.setWeekend("saturday");break;case 17:l.setDateFormat(i[a].substr(11)),this.$=i[a].substr(11);break;case 18:l.enableInclusiveEndDates(),this.$=i[a].substr(18);break;case 19:l.TopAxis(),this.$=i[a].substr(8);break;case 20:l.setAxisFormat(i[a].substr(11)),this.$=i[a].substr(11);break;case 21:l.setTickInterval(i[a].substr(13)),this.$=i[a].substr(13);break;case 22:l.setExcludes(i[a].substr(9)),this.$=i[a].substr(9);break;case 23:l.setIncludes(i[a].substr(9)),this.$=i[a].substr(9);break;case 24:l.setTodayMarker(i[a].substr(12)),this.$=i[a].substr(12);break;case 27:l.setDiagramTitle(i[a].substr(6)),this.$=i[a].substr(6);break;case 28:this.$=i[a].trim(),l.setAccTitle(this.$);break;case 29:case 30:this.$=i[a].trim(),l.setAccDescription(this.$);break;case 31:l.addSection(i[a].substr(8)),this.$=i[a].substr(8);break;case 33:l.addTask(i[a-1],i[a]),this.$="task";break;case 34:this.$=i[a-1],l.setClickEvent(i[a-1],i[a],null);break;case 35:this.$=i[a-2],l.setClickEvent(i[a-2],i[a-1],i[a]);break;case 36:this.$=i[a-2],l.setClickEvent(i[a-2],i[a-1],null),l.setLink(i[a-2],i[a]);break;case 37:this.$=i[a-3],l.setClickEvent(i[a-3],i[a-2],i[a-1]),l.setLink(i[a-3],i[a]);break;case 38:this.$=i[a-2],l.setClickEvent(i[a-2],i[a],null),l.setLink(i[a-2],i[a-1]);break;case 39:this.$=i[a-3],l.setClickEvent(i[a-3],i[a-1],i[a]),l.setLink(i[a-3],i[a-2]);break;case 40:this.$=i[a-1],l.setLink(i[a-1],i[a]);break;case 41:case 47:this.$=i[a-1]+" "+i[a];break;case 42:case 43:case 45:this.$=i[a-2]+" "+i[a-1]+" "+i[a];break;case 44:case 46:this.$=i[a-3]+" "+i[a-2]+" "+i[a-1]+" "+i[a];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:r,13:n,14:o,15:d,16:m,17:$,18:S,19:18,20:D,21:x,22:L,23:_,24:w,25:H,26:E,27:b,28:M,29:A,30:I,31:B,33:z,35:j,36:W,37:24,38:y,40:v},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:36,11:17,12:r,13:n,14:o,15:d,16:m,17:$,18:S,19:18,20:D,21:x,22:L,23:_,24:w,25:H,26:E,27:b,28:M,29:A,30:I,31:B,33:z,35:j,36:W,37:24,38:y,40:v},t(e,[2,5]),t(e,[2,6]),t(e,[2,17]),t(e,[2,18]),t(e,[2,19]),t(e,[2,20]),t(e,[2,21]),t(e,[2,22]),t(e,[2,23]),t(e,[2,24]),t(e,[2,25]),t(e,[2,26]),t(e,[2,27]),{32:[1,37]},{34:[1,38]},t(e,[2,30]),t(e,[2,31]),t(e,[2,32]),{39:[1,39]},t(e,[2,8]),t(e,[2,9]),t(e,[2,10]),t(e,[2,11]),t(e,[2,12]),t(e,[2,13]),t(e,[2,14]),t(e,[2,15]),t(e,[2,16]),{41:[1,40],43:[1,41]},t(e,[2,4]),t(e,[2,28]),t(e,[2,29]),t(e,[2,33]),t(e,[2,34],{42:[1,42],43:[1,43]}),t(e,[2,40],{41:[1,44]}),t(e,[2,35],{43:[1,45]}),t(e,[2,36]),t(e,[2,38],{42:[1,46]}),t(e,[2,37]),t(e,[2,39])],defaultActions:{},parseError:u(function(s,f){if(f.recoverable)this.trace(s);else{var h=Error(s);throw h.hash=f,h}},"parseError"),parse:u(function(s){var f=this,h=[0],l=[],p=[null],i=[],c=this.table,a="",C=0,T=0,Y=0,F=2,O=1,ut=i.slice.call(arguments,1),P=Object.create(this.lexer),X={yy:{}};for(var dt in this.yy)Object.prototype.hasOwnProperty.call(this.yy,dt)&&(X.yy[dt]=this.yy[dt]);P.setInput(s,X.yy),X.yy.lexer=P,X.yy.parser=this,P.yylloc===void 0&&(P.yylloc={});var nt=P.yylloc;i.push(nt);var kt=P.options&&P.options.ranges;typeof X.yy.parseError=="function"?this.parseError=X.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ht(U){h.length-=2*U,p.length-=U,i.length-=U}u(ht,"popStack");function st(){var U=l.pop()||P.lex()||O;return typeof U!="number"&&(U instanceof Array&&(l=U,U=l.pop()),U=f.symbols_[U]||U),U}u(st,"lex");for(var G,tt,R,V,it,q={},et,J,Gt,pt;;){if(R=h[h.length-1],this.defaultActions[R]?V=this.defaultActions[R]:(G??(G=st()),V=c[R]&&c[R][G]),V===void 0||!V.length||!V[0]){var Vt="";for(et in pt=[],c[R])this.terminals_[et]&&et>F&&pt.push("'"+this.terminals_[et]+"'");Vt=P.showPosition?"Parse error on line "+(C+1)+`:
`+P.showPosition()+`
Expecting `+pt.join(", ")+", got '"+(this.terminals_[G]||G)+"'":"Parse error on line "+(C+1)+": Unexpected "+(G==O?"end of input":"'"+(this.terminals_[G]||G)+"'"),this.parseError(Vt,{text:P.match,token:this.terminals_[G]||G,line:P.yylineno,loc:nt,expected:pt})}if(V[0]instanceof Array&&V.length>1)throw Error("Parse Error: multiple actions possible at state: "+R+", token: "+G);switch(V[0]){case 1:h.push(G),p.push(P.yytext),i.push(P.yylloc),h.push(V[1]),G=null,tt?(G=tt,tt=null):(T=P.yyleng,a=P.yytext,C=P.yylineno,nt=P.yylloc,Y>0&&Y--);break;case 2:if(J=this.productions_[V[1]][1],q.$=p[p.length-J],q._$={first_line:i[i.length-(J||1)].first_line,last_line:i[i.length-1].last_line,first_column:i[i.length-(J||1)].first_column,last_column:i[i.length-1].last_column},kt&&(q._$.range=[i[i.length-(J||1)].range[0],i[i.length-1].range[1]]),it=this.performAction.apply(q,[a,T,C,X.yy,V[1],p,i].concat(ut)),it!==void 0)return it;J&&(h=h.slice(0,-1*J*2),p=p.slice(0,-1*J),i=i.slice(0,-1*J)),h.push(this.productions_[V[1]][0]),p.push(q.$),i.push(q._$),Gt=c[h[h.length-2]][h[h.length-1]],h.push(Gt);break;case 3:return!0}}return!0},"parse")};g.lexer=(function(){return{EOF:1,parseError:u(function(s,f){if(this.yy.parser)this.yy.parser.parseError(s,f);else throw Error(s)},"parseError"),setInput:u(function(s,f){return this.yy=f||this.yy||{},this._input=s,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:u(function(){var s=this._input[0];return this.yytext+=s,this.yyleng++,this.offset++,this.match+=s,this.matched+=s,s.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),s},"input"),unput:u(function(s){var f=s.length,h=s.split(/(?:\r\n?|\n)/g);this._input=s+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-f),this.offset-=f;var l=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var p=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===l.length?this.yylloc.first_column:0)+l[l.length-h.length].length-h[0].length:this.yylloc.first_column-f},this.options.ranges&&(this.yylloc.range=[p[0],p[0]+this.yyleng-f]),this.yyleng=this.yytext.length,this},"unput"),more:u(function(){return this._more=!0,this},"more"),reject:u(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:u(function(s){this.unput(this.match.slice(s))},"less"),pastInput:u(function(){var s=this.matched.substr(0,this.matched.length-this.match.length);return(s.length>20?"...":"")+s.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:u(function(){var s=this.match;return s.length<20&&(s+=this._input.substr(0,20-s.length)),(s.substr(0,20)+(s.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:u(function(){var s=this.pastInput(),f=Array(s.length+1).join("-");return s+this.upcomingInput()+`
`+f+"^"},"showPosition"),test_match:u(function(s,f){var h,l,p;if(this.options.backtrack_lexer&&(p={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(p.yylloc.range=this.yylloc.range.slice(0))),l=s[0].match(/(?:\r\n?|\n).*/g),l&&(this.yylineno+=l.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:l?l[l.length-1].length-l[l.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+s[0].length},this.yytext+=s[0],this.match+=s[0],this.matches=s,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(s[0].length),this.matched+=s[0],h=this.performAction.call(this,this.yy,this,f,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var i in p)this[i]=p[i];return!1}return!1},"test_match"),next:u(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var s,f,h,l;this._more||(this.yytext="",this.match="");for(var p=this._currentRules(),i=0;i<p.length;i++)if(h=this._input.match(this.rules[p[i]]),h&&(!f||h[0].length>f[0].length)){if(f=h,l=i,this.options.backtrack_lexer){if(s=this.test_match(h,p[i]),s!==!1)return s;if(this._backtrack){f=!1;continue}else return!1}else if(!this.options.flex)break}return f?(s=this.test_match(f,p[l]),s===!1?!1:s):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:u(function(){return this.next()||this.lex()},"lex"),begin:u(function(s){this.conditionStack.push(s)},"begin"),popState:u(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:u(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:u(function(s){return s=this.conditionStack.length-1-Math.abs(s||0),s>=0?this.conditionStack[s]:"INITIAL"},"topState"),pushState:u(function(s){this.begin(s)},"pushState"),stateStackSize:u(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:u(function(s,f,h,l){switch(h){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}}})();function k(){this.yy={}}return u(k,"Parser"),k.prototype=g,g.Parser=k,new k})();Mt.parser=Mt;var Xe=Mt;Z.default.extend(Ze.default),Z.default.extend(qe.default),Z.default.extend(Qe.default);var ie={friday:5,saturday:6},Q="",Et="",At=void 0,Lt="",ft=[],mt=[],Yt=new Map,It=[],Tt=[],ct="",Ft="",re=["active","done","crit","milestone","vert"],Wt=[],yt=!1,Ot=!1,Pt="sunday",xt="saturday",zt=0,Je=u(function(){It=[],Tt=[],ct="",Wt=[],Nt=0,Ht=void 0,wt=void 0,N=[],Q="",Et="",Ft="",At=void 0,Lt="",ft=[],mt=[],yt=!1,Ot=!1,zt=0,Yt=new Map,Ee(),Pt="sunday",xt="saturday"},"clear"),Ke=u(function(t){Et=t},"setAxisFormat"),ti=u(function(){return Et},"getAxisFormat"),ei=u(function(t){At=t},"setTickInterval"),ii=u(function(){return At},"getTickInterval"),ri=u(function(t){Lt=t},"setTodayMarker"),ni=u(function(){return Lt},"getTodayMarker"),si=u(function(t){Q=t},"setDateFormat"),ai=u(function(){yt=!0},"enableInclusiveEndDates"),oi=u(function(){return yt},"endDatesAreInclusive"),ci=u(function(){Ot=!0},"enableTopAxis"),li=u(function(){return Ot},"topAxisEnabled"),ui=u(function(t){Ft=t},"setDisplayMode"),di=u(function(){return Ft},"getDisplayMode"),hi=u(function(){return Q},"getDateFormat"),fi=u(function(t){ft=t.toLowerCase().split(/[\s,]+/)},"setIncludes"),mi=u(function(){return ft},"getIncludes"),yi=u(function(t){mt=t.toLowerCase().split(/[\s,]+/)},"setExcludes"),ki=u(function(){return mt},"getExcludes"),pi=u(function(){return Yt},"getLinks"),gi=u(function(t){ct=t,It.push(t)},"addSection"),bi=u(function(){return It},"getSections"),vi=u(function(){let t=le(),e=0;for(;!t&&e<10;)t=le(),e++;return Tt=N,Tt},"getTasks"),ne=u(function(t,e,r,n){let o=t.format(e.trim()),d=t.format("YYYY-MM-DD");return n.includes(o)||n.includes(d)?!1:r.includes("weekends")&&(t.isoWeekday()===ie[xt]||t.isoWeekday()===ie[xt]+1)||r.includes(t.format("dddd").toLowerCase())?!0:r.includes(o)||r.includes(d)},"isInvalidDate"),Ti=u(function(t){Pt=t},"setWeekday"),xi=u(function(){return Pt},"getWeekday"),wi=u(function(t){xt=t},"setWeekend"),se=u(function(t,e,r,n){if(!r.length||t.manualEndTime)return;let o;o=t.startTime instanceof Date?(0,Z.default)(t.startTime):(0,Z.default)(t.startTime,e,!0),o=o.add(1,"d");let d;d=t.endTime instanceof Date?(0,Z.default)(t.endTime):(0,Z.default)(t.endTime,e,!0);let[m,$]=_i(o,d,e,r,n);t.endTime=m.toDate(),t.renderEndTime=$},"checkTaskDates"),_i=u(function(t,e,r,n,o){let d=!1,m=null;for(;t<=e;)d||(m=e.toDate()),d=ne(t,r,n,o),d&&(e=e.add(1,"d")),t=t.add(1,"d");return[e,m]},"fixTaskDates"),Bt=u(function(t,e,r){r=r.trim();let n=/^after\s+(?<ids>[\d\w- ]+)/.exec(r);if(n!==null){let d=null;for(let $ of n.groups.ids.split(" ")){let S=rt($);S!==void 0&&(!d||S.endTime>d.endTime)&&(d=S)}if(d)return d.endTime;let m=new Date;return m.setHours(0,0,0,0),m}let o=(0,Z.default)(r,e.trim(),!0);if(o.isValid())return o.toDate();{gt.debug("Invalid date:"+r),gt.debug("With date format:"+e.trim());let d=new Date(r);if(d===void 0||isNaN(d.getTime())||d.getFullYear()<-1e4||d.getFullYear()>1e4)throw Error("Invalid date:"+r);return d}},"getStartDate"),ae=u(function(t){let e=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return e===null?[NaN,"ms"]:[Number.parseFloat(e[1]),e[2]]},"parseDuration"),oe=u(function(t,e,r,n=!1){r=r.trim();let o=/^until\s+(?<ids>[\d\w- ]+)/.exec(r);if(o!==null){let D=null;for(let L of o.groups.ids.split(" ")){let _=rt(L);_!==void 0&&(!D||_.startTime<D.startTime)&&(D=_)}if(D)return D.startTime;let x=new Date;return x.setHours(0,0,0,0),x}let d=(0,Z.default)(r,e.trim(),!0);if(d.isValid())return n&&(d=d.add(1,"d")),d.toDate();let m=(0,Z.default)(t),[$,S]=ae(r);if(!Number.isNaN($)){let D=m.add($,S);D.isValid()&&(m=D)}return m.toDate()},"getEndDate"),Nt=0,lt=u(function(t){return t===void 0?(Nt+=1,"task"+Nt):t},"parseId"),Di=u(function(t,e){let r;r=e.substr(0,1)===":"?e.substr(1,e.length):e;let n=r.split(","),o={};jt(n,o,re);for(let m=0;m<n.length;m++)n[m]=n[m].trim();let d="";switch(n.length){case 1:o.id=lt(),o.startTime=t.endTime,d=n[0];break;case 2:o.id=lt(),o.startTime=Bt(void 0,Q,n[0]),d=n[1];break;case 3:o.id=lt(n[0]),o.startTime=Bt(void 0,Q,n[1]),d=n[2];break;default:}return d&&(o.endTime=oe(o.startTime,Q,d,yt),o.manualEndTime=(0,Z.default)(d,"YYYY-MM-DD",!0).isValid(),se(o,Q,mt,ft)),o},"compileData"),$i=u(function(t,e){let r;r=e.substr(0,1)===":"?e.substr(1,e.length):e;let n=r.split(","),o={};jt(n,o,re);for(let d=0;d<n.length;d++)n[d]=n[d].trim();switch(n.length){case 1:o.id=lt(),o.startTime={type:"prevTaskEnd",id:t},o.endTime={data:n[0]};break;case 2:o.id=lt(),o.startTime={type:"getStartDate",startData:n[0]},o.endTime={data:n[1]};break;case 3:o.id=lt(n[0]),o.startTime={type:"getStartDate",startData:n[1]},o.endTime={data:n[2]};break;default:}return o},"parseData"),Ht,wt,N=[],ce={},Si=u(function(t,e){let r={section:ct,type:ct,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:e},task:t,classes:[]},n=$i(wt,e);r.raw.startTime=n.startTime,r.raw.endTime=n.endTime,r.id=n.id,r.prevTaskId=wt,r.active=n.active,r.done=n.done,r.crit=n.crit,r.milestone=n.milestone,r.vert=n.vert,r.order=zt,zt++;let o=N.push(r);wt=r.id,ce[r.id]=o-1},"addTask"),rt=u(function(t){let e=ce[t];return N[e]},"findTaskById"),Ci=u(function(t,e){let r={section:ct,type:ct,description:t,task:t,classes:[]},n=Di(Ht,e);r.startTime=n.startTime,r.endTime=n.endTime,r.id=n.id,r.active=n.active,r.done=n.done,r.crit=n.crit,r.milestone=n.milestone,r.vert=n.vert,Ht=r,Tt.push(r)},"addTaskOrg"),le=u(function(){let t=u(function(r){let n=N[r],o="";switch(N[r].raw.startTime.type){case"prevTaskEnd":n.startTime=rt(n.prevTaskId).endTime;break;case"getStartDate":o=Bt(void 0,Q,N[r].raw.startTime.startData),o&&(N[r].startTime=o);break}return N[r].startTime&&(N[r].endTime=oe(N[r].startTime,Q,N[r].raw.endTime.data,yt),N[r].endTime&&(N[r].processed=!0,N[r].manualEndTime=(0,Z.default)(N[r].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),se(N[r],Q,mt,ft))),N[r].processed},"compileTask"),e=!0;for(let[r,n]of N.entries())t(r),e&&(e=n.processed);return e},"compileTasks"),Mi=u(function(t,e){let r=e;ot().securityLevel!=="loose"&&(r=(0,Ue.sanitizeUrl)(e)),t.split(",").forEach(function(n){rt(n)!==void 0&&(de(n,()=>{window.open(r,"_self")}),Yt.set(n,r))}),ue(t,"clickable")},"setLink"),ue=u(function(t,e){t.split(",").forEach(function(r){let n=rt(r);n!==void 0&&n.classes.push(e)})},"setClass"),Ei=u(function(t,e,r){if(ot().securityLevel!=="loose"||e===void 0)return;let n=[];if(typeof r=="string"){n=r.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let o=0;o<n.length;o++){let d=n[o].trim();d.startsWith('"')&&d.endsWith('"')&&(d=d.substr(1,d.length-2)),n[o]=d}}n.length===0&&n.push(t),rt(t)!==void 0&&de(t,()=>{De.runFunc(e,...n)})},"setClickFun"),de=u(function(t,e){Wt.push(function(){let r=document.querySelector(`[id="${t}"]`);r!==null&&r.addEventListener("click",function(){e()})},function(){let r=document.querySelector(`[id="${t}-text"]`);r!==null&&r.addEventListener("click",function(){e()})})},"pushFun"),Ai={getConfig:u(()=>ot().gantt,"getConfig"),clear:Je,setDateFormat:si,getDateFormat:hi,enableInclusiveEndDates:ai,endDatesAreInclusive:oi,enableTopAxis:ci,topAxisEnabled:li,setAxisFormat:Ke,getAxisFormat:ti,setTickInterval:ei,getTickInterval:ii,setTodayMarker:ri,getTodayMarker:ni,setAccTitle:$e,getAccTitle:Ye,setDiagramTitle:Ce,getDiagramTitle:Se,setDisplayMode:ui,getDisplayMode:di,setAccDescription:Ie,getAccDescription:Me,addSection:gi,getSections:bi,getTasks:vi,addTask:Si,findTaskById:rt,addTaskOrg:Ci,setIncludes:fi,getIncludes:mi,setExcludes:yi,getExcludes:ki,setClickEvent:u(function(t,e,r){t.split(",").forEach(function(n){Ei(n,e,r)}),ue(t,"clickable")},"setClickEvent"),setLink:Mi,getLinks:pi,bindFunctions:u(function(t){Wt.forEach(function(e){e(t)})},"bindFunctions"),parseDuration:ae,isInvalidDate:ne,setWeekday:Ti,getWeekday:xi,setWeekend:wi};function jt(t,e,r){let n=!0;for(;n;)n=!1,r.forEach(function(o){let d="^\\s*"+o+"\\s*$",m=new RegExp(d);t[0].match(m)&&(e[o]=!0,t.shift(1),n=!0)})}u(jt,"getTaskTags");var Li=u(function(){gt.debug("Something is calling, setConf, remove the call")},"setConf"),he={monday:pe,tuesday:ve,wednesday:be,thursday:Te,friday:we,saturday:ge,sunday:xe},Yi=u((t,e)=>{let r=[...t].map(()=>-1/0),n=[...t].sort((d,m)=>d.startTime-m.startTime||d.order-m.order),o=0;for(let d of n)for(let m=0;m<r.length;m++)if(d.startTime>=r[m]){r[m]=d.endTime,d.order=m+e,m>o&&(o=m);break}return o},"getMaxIntersections"),K,Ii={parser:Xe,db:Ai,renderer:{setConf:Li,draw:u(function(t,e,r,n){let o=ot().gantt,d=ot().securityLevel,m;d==="sandbox"&&(m=Dt("#i"+e));let $=Dt(d==="sandbox"?m.nodes()[0].contentDocument.body:"body"),S=d==="sandbox"?m.nodes()[0].contentDocument:document,D=S.getElementById(e);K=D.parentElement.offsetWidth,K===void 0&&(K=1200),o.useWidth!==void 0&&(K=o.useWidth);let x=n.db.getTasks(),L=[];for(let y of x)L.push(y.type);L=W(L);let _={},w=2*o.topPadding;if(n.db.getDisplayMode()==="compact"||o.displayMode==="compact"){let y={};for(let g of x)y[g.section]===void 0?y[g.section]=[g]:y[g.section].push(g);let v=0;for(let g of Object.keys(y)){let k=Yi(y[g],v)+1;v+=k,w+=k*(o.barHeight+o.barGap),_[g]=k}}else{w+=x.length*(o.barHeight+o.barGap);for(let y of L)_[y]=x.filter(v=>v.type===y).length}D.setAttribute("viewBox","0 0 "+K+" "+w);let H=$.select(`[id="${e}"]`),E=me().domain([ye(x,function(y){return y.startTime}),ke(x,function(y){return y.endTime})]).rangeRound([0,K-o.leftPadding-o.rightPadding]);function b(y,v){let g=y.startTime,k=v.startTime,s=0;return g>k?s=1:g<k&&(s=-1),s}u(b,"taskCompare"),x.sort(b),M(x,K,w),Ae(H,w,K,o.useMaxWidth),H.append("text").text(n.db.getDiagramTitle()).attr("x",K/2).attr("y",o.titleTopMargin).attr("class","titleText");function M(y,v,g){let k=o.barHeight,s=k+o.barGap,f=o.topPadding,h=o.leftPadding,l=fe().domain([0,L.length]).range(["#00B9FA","#F95002"]).interpolate(_e);I(s,f,h,v,g,y,n.db.getExcludes(),n.db.getIncludes()),B(h,f,v,g),A(y,s,f,h,k,l,v,g),z(s,f,h,k,l),j(h,f,v,g)}u(M,"makeGantt");function A(y,v,g,k,s,f,h){y.sort((c,a)=>c.vert===a.vert?0:c.vert?1:-1);let l=[...new Set(y.map(c=>c.order))].map(c=>y.find(a=>a.order===c));H.append("g").selectAll("rect").data(l).enter().append("rect").attr("x",0).attr("y",function(c,a){return a=c.order,a*v+g-2}).attr("width",function(){return h-o.rightPadding/2}).attr("height",v).attr("class",function(c){for(let[a,C]of L.entries())if(c.type===C)return"section section"+a%o.numberSectionStyles;return"section section0"}).enter();let p=H.append("g").selectAll("rect").data(y).enter(),i=n.db.getLinks();if(p.append("rect").attr("id",function(c){return c.id}).attr("rx",3).attr("ry",3).attr("x",function(c){return c.milestone?E(c.startTime)+k+.5*(E(c.endTime)-E(c.startTime))-.5*s:E(c.startTime)+k}).attr("y",function(c,a){return a=c.order,c.vert?o.gridLineStartPadding:a*v+g}).attr("width",function(c){return c.milestone?s:c.vert?.08*s:E(c.renderEndTime||c.endTime)-E(c.startTime)}).attr("height",function(c){return c.vert?x.length*(o.barHeight+o.barGap)+o.barHeight*2:s}).attr("transform-origin",function(c,a){return a=c.order,(E(c.startTime)+k+.5*(E(c.endTime)-E(c.startTime))).toString()+"px "+(a*v+g+.5*s).toString()+"px"}).attr("class",function(c){let a="";c.classes.length>0&&(a=c.classes.join(" "));let C=0;for(let[Y,F]of L.entries())c.type===F&&(C=Y%o.numberSectionStyles);let T="";return c.active?c.crit?T+=" activeCrit":T=" active":c.done?T=c.crit?" doneCrit":" done":c.crit&&(T+=" crit"),T.length===0&&(T=" task"),c.milestone&&(T=" milestone "+T),c.vert&&(T=" vert "+T),T+=C,T+=" "+a,"task"+T}),p.append("text").attr("id",function(c){return c.id+"-text"}).text(function(c){return c.task}).attr("font-size",o.fontSize).attr("x",function(c){let a=E(c.startTime),C=E(c.renderEndTime||c.endTime);if(c.milestone&&(a+=.5*(E(c.endTime)-E(c.startTime))-.5*s,C=a+s),c.vert)return E(c.startTime)+k;let T=this.getBBox().width;return T>C-a?C+T+1.5*o.leftPadding>h?a+k-5:C+k+5:(C-a)/2+a+k}).attr("y",function(c,a){return c.vert?o.gridLineStartPadding+x.length*(o.barHeight+o.barGap)+60:(a=c.order,a*v+o.barHeight/2+(o.fontSize/2-2)+g)}).attr("text-height",s).attr("class",function(c){let a=E(c.startTime),C=E(c.endTime);c.milestone&&(C=a+s);let T=this.getBBox().width,Y="";c.classes.length>0&&(Y=c.classes.join(" "));let F=0;for(let[ut,P]of L.entries())c.type===P&&(F=ut%o.numberSectionStyles);let O="";return c.active&&(O=c.crit?"activeCritText"+F:"activeText"+F),c.done?O=c.crit?O+" doneCritText"+F:O+" doneText"+F:c.crit&&(O=O+" critText"+F),c.milestone&&(O+=" milestoneText"),c.vert&&(O+=" vertText"),T>C-a?C+T+1.5*o.leftPadding>h?Y+" taskTextOutsideLeft taskTextOutside"+F+" "+O:Y+" taskTextOutsideRight taskTextOutside"+F+" "+O+" width-"+T:Y+" taskText taskText"+F+" "+O+" width-"+T}),ot().securityLevel==="sandbox"){let c;c=Dt("#i"+e);let a=c.nodes()[0].contentDocument;p.filter(function(C){return i.has(C.id)}).each(function(C){var T=a.querySelector("#"+C.id),Y=a.querySelector("#"+C.id+"-text");let F=T.parentNode;var O=a.createElement("a");O.setAttribute("xlink:href",i.get(C.id)),O.setAttribute("target","_top"),F.appendChild(O),O.appendChild(T),O.appendChild(Y)})}}u(A,"drawRects");function I(y,v,g,k,s,f,h,l){if(h.length===0&&l.length===0)return;let p,i;for(let{startTime:Y,endTime:F}of f)(p===void 0||Y<p)&&(p=Y),(i===void 0||F>i)&&(i=F);if(!p||!i)return;if((0,Ct.default)(i).diff((0,Ct.default)(p),"year")>5){gt.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}let c=n.db.getDateFormat(),a=[],C=null,T=(0,Ct.default)(p);for(;T.valueOf()<=i;)n.db.isInvalidDate(T,c,h,l)?C?C.end=T:C={start:T,end:T}:C&&(C=(a.push(C),null)),T=T.add(1,"d");H.append("g").selectAll("rect").data(a).enter().append("rect").attr("id",Y=>"exclude-"+Y.start.format("YYYY-MM-DD")).attr("x",Y=>E(Y.start.startOf("day"))+g).attr("y",o.gridLineStartPadding).attr("width",Y=>E(Y.end.endOf("day"))-E(Y.start.startOf("day"))).attr("height",s-v-o.gridLineStartPadding).attr("transform-origin",function(Y,F){return(E(Y.start)+g+.5*(E(Y.end)-E(Y.start))).toString()+"px "+(F*y+.5*s).toString()+"px"}).attr("class","exclude-range")}u(I,"drawExcludeDays");function B(y,v,g,k){let s=n.db.getDateFormat(),f=n.db.getAxisFormat(),h;h=f||(s==="D"?"%d":o.axisFormat??"%Y-%m-%d");let l=je(E).tickSize(-k+v+o.gridLineStartPadding).tickFormat(Qt(h)),p=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(n.db.getTickInterval()||o.tickInterval);if(p!==null){let i=p[1],c=p[2],a=n.db.getWeekday()||o.weekday;switch(c){case"millisecond":l.ticks(Ut.every(i));break;case"second":l.ticks(Zt.every(i));break;case"minute":l.ticks(Rt.every(i));break;case"hour":l.ticks(Jt.every(i));break;case"day":l.ticks(Xt.every(i));break;case"week":l.ticks(he[a].every(i));break;case"month":l.ticks(qt.every(i));break}}if(H.append("g").attr("class","grid").attr("transform","translate("+y+", "+(k-50)+")").call(l).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),n.db.topAxisEnabled()||o.topAxis){let i=He(E).tickSize(-k+v+o.gridLineStartPadding).tickFormat(Qt(h));if(p!==null){let c=p[1],a=p[2],C=n.db.getWeekday()||o.weekday;switch(a){case"millisecond":i.ticks(Ut.every(c));break;case"second":i.ticks(Zt.every(c));break;case"minute":i.ticks(Rt.every(c));break;case"hour":i.ticks(Jt.every(c));break;case"day":i.ticks(Xt.every(c));break;case"week":i.ticks(he[C].every(c));break;case"month":i.ticks(qt.every(c));break}}H.append("g").attr("class","grid").attr("transform","translate("+y+", "+v+")").call(i).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}u(B,"makeGrid");function z(y,v){let g=0,k=Object.keys(_).map(s=>[s,_[s]]);H.append("g").selectAll("text").data(k).enter().append(function(s){let f=s[0].split(Le.lineBreakRegex),h=-(f.length-1)/2,l=S.createElementNS("http://www.w3.org/2000/svg","text");l.setAttribute("dy",h+"em");for(let[p,i]of f.entries()){let c=S.createElementNS("http://www.w3.org/2000/svg","tspan");c.setAttribute("alignment-baseline","central"),c.setAttribute("x","10"),p>0&&c.setAttribute("dy","1em"),c.textContent=i,l.appendChild(c)}return l}).attr("x",10).attr("y",function(s,f){if(f>0)for(let h=0;h<f;h++)return g+=k[f-1][1],s[1]*y/2+g*y+v;else return s[1]*y/2+v}).attr("font-size",o.sectionFontSize).attr("class",function(s){for(let[f,h]of L.entries())if(s[0]===h)return"sectionTitle sectionTitle"+f%o.numberSectionStyles;return"sectionTitle"})}u(z,"vertLabels");function j(y,v,g,k){let s=n.db.getTodayMarker();if(s==="off")return;let f=H.append("g").attr("class","today"),h=new Date,l=f.append("line");l.attr("x1",E(h)+y).attr("x2",E(h)+y).attr("y1",o.titleTopMargin).attr("y2",k-o.titleTopMargin).attr("class","today"),s!==""&&l.attr("style",s.replace(/,/g,";"))}u(j,"drawToday");function W(y){let v={},g=[];for(let k=0,s=y.length;k<s;++k)Object.prototype.hasOwnProperty.call(v,y[k])||(v[y[k]]=!0,g.push(y[k]));return g}u(W,"checkUnique")},"draw")},styles:u(t=>`
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

  .vert {
    stroke: ${t.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${t.vertLineColor} !important;
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
`,"getStyles")};export{Ii as diagram};
