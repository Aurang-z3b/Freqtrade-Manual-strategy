import{Z as V,$ as b,ab as y,c as i,a as o,a2 as $,N as m,H as P,e as f,d as H,r as _,o as M,U as q,w as F,l as v,f as c,s as R,i as U,n as W,k as g,F as x,b as p,g as j,h as d,a9 as O,B as T,_ as Z}from"./index-D5NhhJB3.js";import{_ as J}from"./check-D8C6_NnG.js";import{_ as K}from"./plus-box-outline-Cf2-MgIw.js";var L=V`
    .p-inputgroup,
    .p-inputgroup .p-iconfield,
    .p-inputgroup .p-floatlabel,
    .p-inputgroup .p-iftalabel {
        display: flex;
        align-items: stretch;
        width: 100%;
    }

    .p-inputgroup .p-inputtext,
    .p-inputgroup .p-inputwrapper {
        flex: 1 1 auto;
        width: 1%;
    }

    .p-inputgroupaddon {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: dt('inputgroup.addon.padding');
        background: dt('inputgroup.addon.background');
        color: dt('inputgroup.addon.color');
        border-block-start: 1px solid dt('inputgroup.addon.border.color');
        border-block-end: 1px solid dt('inputgroup.addon.border.color');
        min-width: dt('inputgroup.addon.min.width');
    }

    .p-inputgroupaddon:first-child,
    .p-inputgroupaddon + .p-inputgroupaddon {
        border-inline-start: 1px solid dt('inputgroup.addon.border.color');
    }

    .p-inputgroupaddon:last-child {
        border-inline-end: 1px solid dt('inputgroup.addon.border.color');
    }

    .p-inputgroupaddon:has(.p-button) {
        padding: 0;
        overflow: hidden;
    }

    .p-inputgroupaddon .p-button {
        border-radius: 0;
    }

    .p-inputgroup > .p-component,
    .p-inputgroup > .p-inputwrapper > .p-component,
    .p-inputgroup > .p-iconfield > .p-component,
    .p-inputgroup > .p-floatlabel > .p-component,
    .p-inputgroup > .p-floatlabel > .p-inputwrapper > .p-component,
    .p-inputgroup > .p-iftalabel > .p-component,
    .p-inputgroup > .p-iftalabel > .p-inputwrapper > .p-component {
        border-radius: 0;
        margin: 0;
    }

    .p-inputgroupaddon:first-child,
    .p-inputgroup > .p-component:first-child,
    .p-inputgroup > .p-inputwrapper:first-child > .p-component,
    .p-inputgroup > .p-iconfield:first-child > .p-component,
    .p-inputgroup > .p-floatlabel:first-child > .p-component,
    .p-inputgroup > .p-floatlabel:first-child > .p-inputwrapper > .p-component,
    .p-inputgroup > .p-iftalabel:first-child > .p-component,
    .p-inputgroup > .p-iftalabel:first-child > .p-inputwrapper > .p-component {
        border-start-start-radius: dt('inputgroup.addon.border.radius');
        border-end-start-radius: dt('inputgroup.addon.border.radius');
    }

    .p-inputgroupaddon:last-child,
    .p-inputgroup > .p-component:last-child,
    .p-inputgroup > .p-inputwrapper:last-child > .p-component,
    .p-inputgroup > .p-iconfield:last-child > .p-component,
    .p-inputgroup > .p-floatlabel:last-child > .p-component,
    .p-inputgroup > .p-floatlabel:last-child > .p-inputwrapper > .p-component,
    .p-inputgroup > .p-iftalabel:last-child > .p-component,
    .p-inputgroup > .p-iftalabel:last-child > .p-inputwrapper > .p-component {
        border-start-end-radius: dt('inputgroup.addon.border.radius');
        border-end-end-radius: dt('inputgroup.addon.border.radius');
    }

    .p-inputgroup .p-component:focus,
    .p-inputgroup .p-component.p-focus,
    .p-inputgroup .p-inputwrapper-focus,
    .p-inputgroup .p-component:focus ~ label,
    .p-inputgroup .p-component.p-focus ~ label,
    .p-inputgroup .p-inputwrapper-focus ~ label {
        z-index: 1;
    }

    .p-inputgroup > .p-button:not(.p-button-icon-only) {
        width: auto;
    }

    .p-inputgroup .p-iconfield + .p-iconfield .p-inputtext {
        border-inline-start: 0;
    }
`,Q={root:"p-inputgroup"},X=b.extend({name:"inputgroup",style:L,classes:Q}),Y={name:"BaseInputGroup",extends:y,style:X,provide:function(){return{$pcInputGroup:this,$parentInstance:this}}},ee={name:"InputGroup",extends:Y,inheritAttrs:!1};function te(e,a,r,s,t,n){return o(),i("div",m({class:e.cx("root")},e.ptmi("root")),[$(e.$slots,"default")],16)}ee.render=te;var ne={root:"p-inputgroupaddon"},oe=b.extend({name:"inputgroupaddon",classes:ne}),pe={name:"BaseInputGroupAddon",extends:y,style:oe,provide:function(){return{$pcInputGroupAddon:this,$parentInstance:this}}},re={name:"InputGroupAddon",extends:pe,inheritAttrs:!1};function ie(e,a,r,s,t,n){return o(),i("div",m({class:e.cx("root")},e.ptmi("root")),[$(e.$slots,"default")],16)}re.render=ie;const ae={viewBox:"0 0 24 24",width:"1.2em",height:"1.2em"};function se(e,a){return o(),i("svg",ae,a[0]||(a[0]=[f("path",{fill:"currentColor",d:"M19 21H8V7h11m0-2H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2m-3-4H4a2 2 0 0 0-2 2v14h2V3h12z"},null,-1)]))}const le=P({name:"mdi-content-copy",render:se}),ue={class:"grow"},ke=H({__name:"EditValue",props:{modelValue:{type:String,required:!0},allowEdit:{type:Boolean,default:!1},allowAdd:{type:Boolean,default:!1},allowDuplicate:{type:Boolean,default:!1},editableName:{type:String,required:!0},alignVertical:{type:Boolean,default:!1}},emits:["delete","new","duplicate","rename"],setup(e,{emit:a}){const r=e,s=a,t=_(""),n=_(0);M(()=>{t.value=r.modelValue});function S(){n.value=0,t.value=r.modelValue}function B(){t.value=t.value+" (copy)",n.value=3}function N(){t.value="",n.value=2}q(()=>r.modelValue,()=>{t.value=r.modelValue});function k(){n.value===2?s("new",t.value):n.value===3?s("duplicate",r.modelValue,t.value):s("rename",r.modelValue,t.value),n.value=0}return(w,l)=>{const A=R,C=O,u=j,I=le,z=T,G=K,D=J,E=Z;return o(),i("form",{class:"flex flex-row",onSubmit:F(k,["prevent"])},[f("div",ue,[c(n)===0?$(w.$slots,"default",{key:0}):(o(),v(A,{key:1,modelValue:c(t),"onUpdate:modelValue":l[0]||(l[0]=h=>U(t)?t.value=h:null),size:"small",fluid:""},null,8,["modelValue"]))]),f("div",{class:W(["mt-auto flex gap-1 ms-1",e.alignVertical?"flex-col":"flex-row"])},[e.allowEdit&&c(n)===0?(o(),i(x,{key:0},[p(u,{size:"small",severity:"secondary",title:`Edit this ${e.editableName}.`,onClick:l[1]||(l[1]=h=>n.value=1)},{icon:d(()=>[p(C)]),_:1},8,["title"]),e.allowDuplicate?(o(),v(u,{key:0,size:"small",severity:"secondary",title:`Duplicate ${e.editableName}.`,onClick:B},{icon:d(()=>[p(I)]),_:1},8,["title"])):g("",!0),p(u,{size:"small",severity:"secondary",title:`Delete this ${e.editableName}.`,onClick:l[2]||(l[2]=h=>w.$emit("delete",e.modelValue))},{icon:d(()=>[p(z)]),_:1},8,["title"])],64)):g("",!0),e.allowAdd&&c(n)===0?(o(),v(u,{key:1,size:"small",title:`Add new ${e.editableName}.`,severity:"primary",onClick:N},{icon:d(()=>[p(G)]),_:1},8,["title"])):g("",!0),c(n)!==0?(o(),i(x,{key:2},[p(u,{size:"small",title:`Add new ${e.editableName}`,severity:"primary",onClick:k},{icon:d(()=>[p(D)]),_:1},8,["title"]),p(u,{size:"small",title:"Abort",severity:"secondary",onClick:S},{icon:d(()=>[p(E)]),_:1})],64)):g("",!0)],2)],32)}}});var de=V`
    .p-progressspinner {
        position: relative;
        margin: 0 auto;
        width: 100px;
        height: 100px;
        display: inline-block;
    }

    .p-progressspinner::before {
        content: '';
        display: block;
        padding-top: 100%;
    }

    .p-progressspinner-spin {
        height: 100%;
        transform-origin: center center;
        width: 100%;
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        right: 0;
        margin: auto;
        animation: p-progressspinner-rotate 2s linear infinite;
    }

    .p-progressspinner-circle {
        stroke-dasharray: 89, 200;
        stroke-dashoffset: 0;
        stroke: dt('progressspinner.colorOne');
        animation:
            p-progressspinner-dash 1.5s ease-in-out infinite,
            p-progressspinner-color 6s ease-in-out infinite;
        stroke-linecap: round;
    }

    @keyframes p-progressspinner-rotate {
        100% {
            transform: rotate(360deg);
        }
    }
    @keyframes p-progressspinner-dash {
        0% {
            stroke-dasharray: 1, 200;
            stroke-dashoffset: 0;
        }
        50% {
            stroke-dasharray: 89, 200;
            stroke-dashoffset: -35px;
        }
        100% {
            stroke-dasharray: 89, 200;
            stroke-dashoffset: -124px;
        }
    }
    @keyframes p-progressspinner-color {
        100%,
        0% {
            stroke: dt('progressspinner.color.one');
        }
        40% {
            stroke: dt('progressspinner.color.two');
        }
        66% {
            stroke: dt('progressspinner.color.three');
        }
        80%,
        90% {
            stroke: dt('progressspinner.color.four');
        }
    }
`,ce={root:"p-progressspinner",spin:"p-progressspinner-spin",circle:"p-progressspinner-circle"},me=b.extend({name:"progressspinner",style:de,classes:ce}),ge={name:"BaseProgressSpinner",extends:y,props:{strokeWidth:{type:String,default:"2"},fill:{type:String,default:"none"},animationDuration:{type:String,default:"2s"}},style:me,provide:function(){return{$pcProgressSpinner:this,$parentInstance:this}}},fe={name:"ProgressSpinner",extends:ge,inheritAttrs:!1,computed:{svgStyle:function(){return{"animation-duration":this.animationDuration}}}},he=["fill","stroke-width"];function ve(e,a,r,s,t,n){return o(),i("div",m({class:e.cx("root"),role:"progressbar"},e.ptmi("root")),[(o(),i("svg",m({class:e.cx("spin"),viewBox:"25 25 50 50",style:n.svgStyle},e.ptm("spin")),[f("circle",m({class:e.cx("circle"),cx:"50",cy:"50",r:"20",fill:e.fill,"stroke-width":e.strokeWidth,strokeMiterlimit:"10"},e.ptm("circle")),null,16,he)],16))],16)}fe.render=ve;export{ke as _,re as a,fe as b,le as c,ee as s};
//# sourceMappingURL=index-aq9KRS0H.js.map
