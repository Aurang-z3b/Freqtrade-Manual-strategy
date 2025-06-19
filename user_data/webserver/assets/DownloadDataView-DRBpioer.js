import{_ as q}from"./DraggableContainer.vue_vue_type_script_setup_true_lang-ByDD7cAj.js";import{_ as G}from"./ExchangeSelect.vue_vue_type_script_setup_true_lang-Dye_toWx.js";import{Z as K,$ as Q,ab as j,ag as ee,c as i,a,k as V,N as M,a2 as te,x as f,z as $,H as se,e as t,d as I,ca as ae,l as v,F as A,m as J,b as o,f as r,g as W,h as p,B as ne,j as re,r as b,u as oe,R as ie,S as le,i as N,W as de,a3 as ue,Y as R,V as F,c3 as me,X as L,t as ce}from"./index-D5NhhJB3.js";import{_ as pe,a as ge}from"./chevron-up-BFJjdW3C.js";import{s as _e}from"./index-D9hPgAiM.js";import{_ as fe}from"./TimeRangeSelect.vue_vue_type_script_setup_true_lang-CVHJxKe3.js";import{_ as be}from"./check-D8C6_NnG.js";import"./plus-box-outline-Cf2-MgIw.js";var ve=K`
    .p-progressbar {
        position: relative;
        overflow: hidden;
        height: dt('progressbar.height');
        background: dt('progressbar.background');
        border-radius: dt('progressbar.border.radius');
    }

    .p-progressbar-value {
        margin: 0;
        background: dt('progressbar.value.background');
    }

    .p-progressbar-label {
        color: dt('progressbar.label.color');
        font-size: dt('progressbar.label.font.size');
        font-weight: dt('progressbar.label.font.weight');
    }

    .p-progressbar-determinate .p-progressbar-value {
        height: 100%;
        width: 0%;
        position: absolute;
        display: none;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        transition: width 1s ease-in-out;
    }

    .p-progressbar-determinate .p-progressbar-label {
        display: inline-flex;
    }

    .p-progressbar-indeterminate .p-progressbar-value::before {
        content: '';
        position: absolute;
        background: inherit;
        inset-block-start: 0;
        inset-inline-start: 0;
        inset-block-end: 0;
        will-change: inset-inline-start, inset-inline-end;
        animation: p-progressbar-indeterminate-anim 2.1s cubic-bezier(0.65, 0.815, 0.735, 0.395) infinite;
    }

    .p-progressbar-indeterminate .p-progressbar-value::after {
        content: '';
        position: absolute;
        background: inherit;
        inset-block-start: 0;
        inset-inline-start: 0;
        inset-block-end: 0;
        will-change: inset-inline-start, inset-inline-end;
        animation: p-progressbar-indeterminate-anim-short 2.1s cubic-bezier(0.165, 0.84, 0.44, 1) infinite;
        animation-delay: 1.15s;
    }

    @keyframes p-progressbar-indeterminate-anim {
        0% {
            inset-inline-start: -35%;
            inset-inline-end: 100%;
        }
        60% {
            inset-inline-start: 100%;
            inset-inline-end: -90%;
        }
        100% {
            inset-inline-start: 100%;
            inset-inline-end: -90%;
        }
    }
    @-webkit-keyframes p-progressbar-indeterminate-anim {
        0% {
            inset-inline-start: -35%;
            inset-inline-end: 100%;
        }
        60% {
            inset-inline-start: 100%;
            inset-inline-end: -90%;
        }
        100% {
            inset-inline-start: 100%;
            inset-inline-end: -90%;
        }
    }

    @keyframes p-progressbar-indeterminate-anim-short {
        0% {
            inset-inline-start: -200%;
            inset-inline-end: 100%;
        }
        60% {
            inset-inline-start: 107%;
            inset-inline-end: -8%;
        }
        100% {
            inset-inline-start: 107%;
            inset-inline-end: -8%;
        }
    }
    @-webkit-keyframes p-progressbar-indeterminate-anim-short {
        0% {
            inset-inline-start: -200%;
            inset-inline-end: 100%;
        }
        60% {
            inset-inline-start: 107%;
            inset-inline-end: -8%;
        }
        100% {
            inset-inline-start: 107%;
            inset-inline-end: -8%;
        }
    }
`,xe={root:function(d){var u=d.instance;return["p-progressbar p-component",{"p-progressbar-determinate":u.determinate,"p-progressbar-indeterminate":u.indeterminate}]},value:"p-progressbar-value",label:"p-progressbar-label"},he=Q.extend({name:"progressbar",style:ve,classes:xe}),ke={name:"BaseProgressBar",extends:j,props:{value:{type:Number,default:null},mode:{type:String,default:"determinate"},showValue:{type:Boolean,default:!0}},style:he,provide:function(){return{$pcProgressBar:this,$parentInstance:this}}},X={name:"ProgressBar",extends:ke,inheritAttrs:!1,computed:{progressStyle:function(){return{width:this.value+"%",display:"flex"}},indeterminate:function(){return this.mode==="indeterminate"},determinate:function(){return this.mode==="determinate"},dataP:function(){return ee({determinate:this.determinate,indeterminate:this.indeterminate})}}},ye=["aria-valuenow","data-p"],we=["data-p"],Se=["data-p"],Ve=["data-p"];function $e(s,d,u,y,m,g){return a(),i("div",M({role:"progressbar",class:s.cx("root"),"aria-valuemin":"0","aria-valuenow":s.value,"aria-valuemax":"100","data-p":g.dataP},s.ptmi("root")),[g.determinate?(a(),i("div",M({key:0,class:s.cx("value"),style:g.progressStyle,"data-p":g.dataP},s.ptm("value")),[s.value!=null&&s.value!==0&&s.showValue?(a(),i("div",M({key:0,class:s.cx("label"),"data-p":g.dataP},s.ptm("label")),[te(s.$slots,"default",{},function(){return[f($(s.value+"%"),1)]})],16,Se)):V("",!0)],16,we)):g.indeterminate?(a(),i("div",M({key:1,class:s.cx("value"),"data-p":g.dataP},s.ptm("value")),null,16,Ve)):V("",!0)],16,ye)}X.render=$e;const Te={viewBox:"0 0 24 24",width:"1.2em",height:"1.2em"};function De(s,d){return a(),i("svg",Te,d[0]||(d[0]=[t("path",{fill:"currentColor",d:"M8 17v-2h8v2zm8-7l-4 4l-4-4h2.5V7h3v3zM5 3h14a2 2 0 0 1 2 2v14c0 1.11-.89 2-2 2H5a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2m0 2v14h14V5z"},null,-1)]))}const Be=se({name:"mdi-download-box-outline",render:De}),Ce={class:"flex flex-row items-end gap-1"},Pe={class:"ms-2 w-full grow space-y-1"},Ee=["title"],Ue={key:1},ze={class:"flex justify-between"},Me={key:1},Ne={key:2,class:"w-25"},Oe={key:3,class:"flex flex-col md:flex-row w-full grow gap-2"},Ae=I({__name:"BackgroundJobTracking",setup(s){const{runningJobs:d,clearJobs:u}=ae();return(y,m)=>{const g=Be,_=be,x=X,w=ne,h=W;return a(),i("div",Ce,[t("ul",Pe,[(a(!0),i(A,null,J(r(d),(c,B)=>{var l,e,C,T,S,k,P,E,U,z;return a(),i("li",{key:B,class:"border p-1 pb-2 rounded-sm dark:border-surface-700 border-surface-300 flex gap-2 items-center",title:B},[((l=c.taskStatus)==null?void 0:l.job_category)==="download_data"?(a(),v(g,{key:0})):(a(),i("span",Ue,$((e=c.taskStatus)==null?void 0:e.job_category),1)),t("div",ze,[((C=c.taskStatus)==null?void 0:C.status)==="success"?(a(),v(_,{key:0,class:"text-success",title:""})):(a(),i("span",Me,$((T=c.taskStatus)==null?void 0:T.status),1)),(S=c.taskStatus)!=null&&S.progress?(a(),i("span",Ne,$((k=c.taskStatus)==null?void 0:k.progress),1)):V("",!0)]),(P=c.taskStatus)!=null&&P.progress?(a(),v(x,{key:2,class:"w-full grow",value:((E=c.taskStatus)==null?void 0:E.progress)/100*100,"show-progress":"",max:100,striped:""},null,8,["value"])):V("",!0),(U=c.taskStatus)!=null&&U.progress_tasks?(a(),i("div",Oe,[(a(!0),i(A,null,J(Object.entries((z=c.taskStatus)==null?void 0:z.progress_tasks),([O,D])=>(a(),i("div",{key:O,class:"w-full"},[f($(D.description)+" ",1),o(x,{class:"w-full grow",value:Math.round(D.progress/D.total*100*100)/100,"show-progress":"",pt:{value:{class:c.taskStatus.status==="success"?"bg-emerald-500":"bg-amber-500"}},striped:""},null,8,["value","pt"])]))),128))])):V("",!0)],8,Ee)}),128))]),Object.keys(r(d)).length>0?(a(),v(h,{key:0,severity:"secondary",class:"ms-auto",onClick:r(u)},{icon:p(()=>[o(w)]),_:1},8,["onClick"])):V("",!0)])}}}),Je=b([{description:"All USDT Pairs",pairs:[".*/USDT"]},{description:"All USDT Futures Pairs",pairs:[".*/USDT:USDT"]}]);function He(){return{pairTemplates:re(()=>Je.value.map((s,d)=>({...s,idx:d})))}}const Re={class:"px-1 mx-auto w-full max-w-4xl lg:max-w-7xl"},Fe={class:"flex mb-3 gap-3 flex-col"},Le={class:"flex flex-col gap-3"},Ie={class:"flex flex-col lg:flex-row gap-3"},We={class:"flex-fill"},Xe={class:"flex flex-col gap-2"},Ye={class:"flex gap-2"},Ze={class:"flex flex-col gap-1"},qe={class:"flex flex-col gap-1"},Ge={class:"flex-fill px-3"},Ke={class:"flex flex-col gap-2"},Qe={class:"px-3 border dark:border-surface-700 border-surface-300 p-2 rounded-sm"},je={class:"flex flex-col gap-2"},et={class:"flex justify-between items-center"},tt={key:0},st={key:1,class:"flex items-center gap-2"},at={class:"mb-2 border dark:border-surface-700 border-surface-300 rounded-sm p-2 text-start"},nt={class:"mb-2 border dark:border-surface-700 border-surface-300 rounded-md p-2 text-start"},rt={class:"mb-2 border dark:border-surface-700 border-surface-300 rounded-md p-2 text-start"},ot={class:"px-3"},it=I({__name:"DownloadDataMain",setup(s){const d=oe(),u=b(["BTC/USDT","ETH/USDT",""]),y=b(["5m","1h"]),m=b({useCustomTimerange:!1,timerange:"",days:30}),{pairTemplates:g}=He(),_=b({customExchange:!1,selectedExchange:{exchange:"binance",trade_mode:{margin_mode:le.NONE,trading_mode:ie.SPOT}}}),x=b(!1),w=b(!1),h=b(!1);function c(l){u.value.push(...l)}async function B(){const l={pairs:u.value.filter(e=>e!==""),timeframes:y.value.filter(e=>e!=="")};m.value.useCustomTimerange&&m.value.timerange?l.timerange=m.value.timerange:l.days=m.value.days,h.value&&(l.erase=x.value,l.download_trades=w.value,_.value.customExchange&&(l.exchange=_.value.selectedExchange.exchange,l.trading_mode=_.value.selectedExchange.trade_mode.trading_mode,l.margin_mode=_.value.selectedExchange.trade_mode.margin_mode)),await d.activeBot.startDataDownload(l)}return(l,e)=>{const C=Ae,T=pe,S=W,k=de,P=fe,E=_e,U=ue,z=ge,O=me,D=G,Y=q;return a(),i("div",Re,[o(C,{class:"mb-4"}),o(Y,{header:"Downloading Data",class:"mx-1 p-4"},{default:p(()=>[t("div",Fe,[t("div",Le,[t("div",Ie,[t("div",We,[t("div",Xe,[e[10]||(e[10]=t("div",{class:"flex justify-between"},[t("h4",{class:"text-start font-bold text-lg"},"Select Pairs"),t("h5",{class:"text-start font-bold text-lg"},"Pairs from template")],-1)),t("div",Ye,[o(T,{modelValue:r(u),"onUpdate:modelValue":e[0]||(e[0]=n=>N(u)?u.value=n:null),placeholder:"Pair",size:"small",class:"flex-grow-1"},null,8,["modelValue"]),t("div",Ze,[t("div",qe,[(a(!0),i(A,null,J(r(g),n=>(a(),v(S,{key:n.idx,severity:"secondary",title:n.pairs.reduce((H,Z)=>`${H}${Z}
`,""),onClick:H=>c(n.pairs)},{default:p(()=>[f($(n.description),1)]),_:2},1032,["title","onClick"]))),128))])])])])]),t("div",Ge,[t("div",Ke,[e[11]||(e[11]=t("h4",{class:"text-start font-bold text-lg"},"Select timeframes",-1)),o(T,{modelValue:r(y),"onUpdate:modelValue":e[1]||(e[1]=n=>N(y)?y.value=n:null),placeholder:"Timeframe"},null,8,["modelValue"])])])]),t("div",Qe,[t("div",je,[t("div",et,[e[13]||(e[13]=t("h4",{class:"text-start mb-0 font-bold text-lg"},"Time Selection",-1)),o(k,{modelValue:r(m).useCustomTimerange,"onUpdate:modelValue":e[2]||(e[2]=n=>r(m).useCustomTimerange=n),class:"mb-0",switch:""},{default:p(()=>e[12]||(e[12]=[f(" Use custom timerange ")])),_:1},8,["modelValue"])]),r(m).useCustomTimerange?(a(),i("div",tt,[o(P,{modelValue:r(m).timerange,"onUpdate:modelValue":e[3]||(e[3]=n=>r(m).timerange=n)},null,8,["modelValue"])])):(a(),i("div",st,[e[14]||(e[14]=t("label",null,"Days to download:",-1)),o(E,{modelValue:r(m).days,"onUpdate:modelValue":e[4]||(e[4]=n=>r(m).days=n),type:"number","aria-label":"Days to download",min:1,step:1,size:"small"},null,8,["modelValue"])]))])]),t("div",at,[o(S,{class:"mb-2",severity:"secondary",onClick:e[5]||(e[5]=n=>h.value=!r(h))},{default:p(()=>[e[15]||(e[15]=f(" Advanced Options ")),r(h)?(a(),v(z,{key:1})):(a(),v(U,{key:0}))]),_:1}),o(R,null,{default:p(()=>[F(t("div",null,[o(O,{severity:"info",class:"mb-2 py-2"},{default:p(()=>e[16]||(e[16]=[f(" Advanced options (Erase data, Download trades, and Custom Exchange settings) will only be applied when this section is expanded. ")])),_:1}),t("div",nt,[o(k,{modelValue:r(x),"onUpdate:modelValue":e[6]||(e[6]=n=>N(x)?x.value=n:null),class:"mb-2"},{default:p(()=>e[17]||(e[17]=[f("Erase existing data")])),_:1},8,["modelValue"]),o(k,{modelValue:r(w),"onUpdate:modelValue":e[7]||(e[7]=n=>N(w)?w.value=n:null),class:"mb-2"},{default:p(()=>e[18]||(e[18]=[f(" Download Trades instead of OHLCV data ")])),_:1},8,["modelValue"])]),t("div",rt,[o(k,{modelValue:r(_).customExchange,"onUpdate:modelValue":e[8]||(e[8]=n=>r(_).customExchange=n),class:"mb-2"},{default:p(()=>e[19]||(e[19]=[f(" Custom Exchange ")])),_:1},8,["modelValue"]),o(R,{name:"fade"},{default:p(()=>[F(o(D,{modelValue:r(_).selectedExchange,"onUpdate:modelValue":e[9]||(e[9]=n=>r(_).selectedExchange=n)},null,8,["modelValue"]),[[L,r(_).customExchange]])]),_:1})])],512),[[L,r(h)]])]),_:1})]),t("div",ot,[o(S,{severity:"primary",onClick:B},{default:p(()=>e[20]||(e[20]=[f("Start Download")])),_:1})])])])]),_:1})])}}}),lt={};function dt(s,d){const u=it;return a(),v(u,{class:"pt-4"})}const vt=ce(lt,[["render",dt]]);export{vt as default};
//# sourceMappingURL=DownloadDataView-DRBpioer.js.map
