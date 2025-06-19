import{Z as j,$ as T,ab as w,c as u,a as c,a2 as h,N as d,ac as E,F as N,k as $,V as C,X as U,l as k,h as R,ad as _,n as P,H as q,e as m,d as Z,r as G,u as X,j as J,A as z,G as Q,b as K,s as Y,f as v,i as tt,m as et,E as at,x as rt,z as nt,t as st,ae as it,af as F,ag as D,ah as S,ai as ot,aj as V,ak as lt,al as B,am as ct,an as O,ao as W,ap as dt,aq as x}from"./index-D5NhhJB3.js";import{a as ut,b as bt}from"./InfoBox.vue_vue_type_script_setup_true_lang-TGv6XaZ6.js";import{b as pt}from"./index-CYBrLTd2.js";var ft=j`
    .p-tabs {
        display: flex;
        flex-direction: column;
    }

    .p-tablist {
        display: flex;
        position: relative;
    }

    .p-tabs-scrollable > .p-tablist {
        overflow: hidden;
    }

    .p-tablist-viewport {
        overflow-x: auto;
        overflow-y: hidden;
        scroll-behavior: smooth;
        scrollbar-width: none;
        overscroll-behavior: contain auto;
    }

    .p-tablist-viewport::-webkit-scrollbar {
        display: none;
    }

    .p-tablist-tab-list {
        position: relative;
        display: flex;
        background: dt('tabs.tablist.background');
        border-style: solid;
        border-color: dt('tabs.tablist.border.color');
        border-width: dt('tabs.tablist.border.width');
    }

    .p-tablist-content {
        flex-grow: 1;
    }

    .p-tablist-nav-button {
        all: unset;
        position: absolute !important;
        flex-shrink: 0;
        inset-block-start: 0;
        z-index: 2;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: dt('tabs.nav.button.background');
        color: dt('tabs.nav.button.color');
        width: dt('tabs.nav.button.width');
        transition:
            color dt('tabs.transition.duration'),
            outline-color dt('tabs.transition.duration'),
            box-shadow dt('tabs.transition.duration');
        box-shadow: dt('tabs.nav.button.shadow');
        outline-color: transparent;
        cursor: pointer;
    }

    .p-tablist-nav-button:focus-visible {
        z-index: 1;
        box-shadow: dt('tabs.nav.button.focus.ring.shadow');
        outline: dt('tabs.nav.button.focus.ring.width') dt('tabs.nav.button.focus.ring.style') dt('tabs.nav.button.focus.ring.color');
        outline-offset: dt('tabs.nav.button.focus.ring.offset');
    }

    .p-tablist-nav-button:hover {
        color: dt('tabs.nav.button.hover.color');
    }

    .p-tablist-prev-button {
        inset-inline-start: 0;
    }

    .p-tablist-next-button {
        inset-inline-end: 0;
    }

    .p-tablist-prev-button:dir(rtl),
    .p-tablist-next-button:dir(rtl) {
        transform: rotate(180deg);
    }

    .p-tab {
        flex-shrink: 0;
        cursor: pointer;
        user-select: none;
        position: relative;
        border-style: solid;
        white-space: nowrap;
        gap: dt('tabs.tab.gap');
        background: dt('tabs.tab.background');
        border-width: dt('tabs.tab.border.width');
        border-color: dt('tabs.tab.border.color');
        color: dt('tabs.tab.color');
        padding: dt('tabs.tab.padding');
        font-weight: dt('tabs.tab.font.weight');
        transition:
            background dt('tabs.transition.duration'),
            border-color dt('tabs.transition.duration'),
            color dt('tabs.transition.duration'),
            outline-color dt('tabs.transition.duration'),
            box-shadow dt('tabs.transition.duration');
        margin: dt('tabs.tab.margin');
        outline-color: transparent;
    }

    .p-tab:not(.p-disabled):focus-visible {
        z-index: 1;
        box-shadow: dt('tabs.tab.focus.ring.shadow');
        outline: dt('tabs.tab.focus.ring.width') dt('tabs.tab.focus.ring.style') dt('tabs.tab.focus.ring.color');
        outline-offset: dt('tabs.tab.focus.ring.offset');
    }

    .p-tab:not(.p-tab-active):not(.p-disabled):hover {
        background: dt('tabs.tab.hover.background');
        border-color: dt('tabs.tab.hover.border.color');
        color: dt('tabs.tab.hover.color');
    }

    .p-tab-active {
        background: dt('tabs.tab.active.background');
        border-color: dt('tabs.tab.active.border.color');
        color: dt('tabs.tab.active.color');
    }

    .p-tabpanels {
        background: dt('tabs.tabpanel.background');
        color: dt('tabs.tabpanel.color');
        padding: dt('tabs.tabpanel.padding');
        outline: 0 none;
    }

    .p-tabpanel:focus-visible {
        box-shadow: dt('tabs.tabpanel.focus.ring.shadow');
        outline: dt('tabs.tabpanel.focus.ring.width') dt('tabs.tabpanel.focus.ring.style') dt('tabs.tabpanel.focus.ring.color');
        outline-offset: dt('tabs.tabpanel.focus.ring.offset');
    }

    .p-tablist-active-bar {
        z-index: 1;
        display: block;
        position: absolute;
        inset-block-end: dt('tabs.active.bar.bottom');
        height: dt('tabs.active.bar.height');
        background: dt('tabs.active.bar.background');
        transition: 250ms cubic-bezier(0.35, 0, 0.25, 1);
    }
`,vt={root:function(t){var r=t.props;return["p-tabs p-component",{"p-tabs-scrollable":r.scrollable}]}},ht=T.extend({name:"tabs",style:ft,classes:vt}),mt={name:"BaseTabs",extends:w,props:{value:{type:[String,Number],default:void 0},lazy:{type:Boolean,default:!1},scrollable:{type:Boolean,default:!1},showNavigators:{type:Boolean,default:!0},tabindex:{type:Number,default:0},selectOnFocus:{type:Boolean,default:!1}},style:ht,provide:function(){return{$pcTabs:this,$parentInstance:this}}},gt={name:"Tabs",extends:mt,inheritAttrs:!1,emits:["update:value"],data:function(){return{d_value:this.value}},watch:{value:function(t){this.d_value=t}},methods:{updateValue:function(t){this.d_value!==t&&(this.d_value=t,this.$emit("update:value",t))},isVertical:function(){return this.orientation==="vertical"}}};function yt(e,t,r,n,o,a){return c(),u("div",d({class:e.cx("root")},e.ptmi("root")),[h(e.$slots,"default")],16)}gt.render=yt;var $t={root:"p-tabpanels"},kt=T.extend({name:"tabpanels",classes:$t}),Tt={name:"BaseTabPanels",extends:w,props:{},style:kt,provide:function(){return{$pcTabPanels:this,$parentInstance:this}}},wt={name:"TabPanels",extends:Tt,inheritAttrs:!1};function Bt(e,t,r,n,o,a){return c(),u("div",d({class:e.cx("root"),role:"presentation"},e.ptmi("root")),[h(e.$slots,"default")],16)}wt.render=Bt;var xt={root:function(t){var r=t.instance;return["p-tabpanel",{"p-tabpanel-active":r.active}]}},Ct=T.extend({name:"tabpanel",classes:xt}),_t={name:"BaseTabPanel",extends:w,props:{value:{type:[String,Number],default:void 0},as:{type:[String,Object],default:"DIV"},asChild:{type:Boolean,default:!1},header:null,headerStyle:null,headerClass:null,headerProps:null,headerActionProps:null,contentStyle:null,contentClass:null,contentProps:null,disabled:Boolean},style:Ct,provide:function(){return{$pcTabPanel:this,$parentInstance:this}}},Pt={name:"TabPanel",extends:_t,inheritAttrs:!1,inject:["$pcTabs"],computed:{active:function(){var t;return E((t=this.$pcTabs)===null||t===void 0?void 0:t.d_value,this.value)},id:function(){var t;return"".concat((t=this.$pcTabs)===null||t===void 0?void 0:t.$id,"_tabpanel_").concat(this.value)},ariaLabelledby:function(){var t;return"".concat((t=this.$pcTabs)===null||t===void 0?void 0:t.$id,"_tab_").concat(this.value)},attrs:function(){return d(this.a11yAttrs,this.ptmi("root",this.ptParams))},a11yAttrs:function(){var t;return{id:this.id,tabindex:(t=this.$pcTabs)===null||t===void 0?void 0:t.tabindex,role:"tabpanel","aria-labelledby":this.ariaLabelledby,"data-pc-name":"tabpanel","data-p-active":this.active}},ptParams:function(){return{context:{active:this.active}}}}};function Lt(e,t,r,n,o,a){var s,i;return a.$pcTabs?(c(),u(N,{key:1},[e.asChild?h(e.$slots,"default",{key:1,class:P(e.cx("root")),active:a.active,a11yAttrs:a.a11yAttrs}):(c(),u(N,{key:0},[!((s=a.$pcTabs)!==null&&s!==void 0&&s.lazy)||a.active?C((c(),k(_(e.as),d({key:0,class:e.cx("root")},a.attrs),{default:R(function(){return[h(e.$slots,"default")]}),_:3},16,["class"])),[[U,(i=a.$pcTabs)!==null&&i!==void 0&&i.lazy?!0:a.active]]):$("",!0)],64))],64)):h(e.$slots,"default",{key:0})}Pt.render=Lt;const At={viewBox:"0 0 24 24",width:"1.2em",height:"1.2em"};function St(e,t){return c(),u("svg",At,t[0]||(t[0]=[m("path",{fill:"currentColor",d:"M12 17a2 2 0 0 0 2-2a2 2 0 0 0-2-2a2 2 0 0 0-2 2a2 2 0 0 0 2 2m6-9a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V10a2 2 0 0 1 2-2h1V6a5 5 0 0 1 5-5a5 5 0 0 1 5 5v2zm-6-5a3 3 0 0 0-3 3v2h6V6a3 3 0 0 0-3-3"},null,-1)]))}const Nt=q({name:"mdi-lock",render:St}),Vt={class:"divide-y divide-surface-300 dark:divide-surface-700 divide-solid border-x border-y rounded-sm border-surface-300 dark:border-surface-700"},It=["title","onClick"],zt=["title"],Kt=Z({__name:"PairSummary",props:{pairlist:{required:!0,type:Array},currentLocks:{required:!1,type:Array,default:()=>[]},trades:{required:!0,type:Array},sortMethod:{default:"normal",type:String},backtestMode:{required:!1,default:!1,type:Boolean},startingBalance:{required:!1,type:Number,default:0}},setup(e){const t=G(""),r=e,n=X(),o=J(()=>{const a=[];return r.pairlist.forEach(s=>{const i=r.trades.filter(f=>f.pair===s),b=r.currentLocks.filter(f=>f.pair===s);let g="",p;b.sort((f,H)=>f.lock_end_timestamp>H.lock_end_timestamp?-1:1),b.length>0&&([p]=b,g=`${z(p.lock_end_timestamp)} - ${p.side} - ${p.reason}`);let l="",y=0,L=0;i.forEach(f=>{y+=f.profit_ratio,L+=f.profit_abs??0}),r.sortMethod=="profit"&&r.startingBalance&&(y=L/r.startingBalance);const I=i.length,A=I?i[0]:void 0;i.length>0&&(l=`Current profit: ${Q(y)}`),A&&(l+=`
Open since: ${z(A.open_timestamp)}`),(t.value===""||s.toLocaleLowerCase().includes(t.value.toLocaleLowerCase()))&&a.push({pair:s,trade:A,locks:p,lockReason:g,profitString:l,profit:y,profitAbs:L,tradeCount:I})}),r.sortMethod==="profit"?a.sort((s,i)=>s.profit>i.profit?-1:1):a.sort((s,i)=>s.trade&&!i.trade?-1:s.trade&&i.trade?s.trade.trade_id>i.trade.trade_id?1:-1:!s.locks&&i.locks?-1:s.locks&&i.locks?s.locks.lock_end_timestamp>i.locks.lock_end_timestamp?1:-1:1),a});return(a,s)=>{const i=Y,b=Nt,g=ut,p=bt;return c(),u("div",null,[m("div",{"label-for":"trade-filter",class:P(["mb-2",{"me-4":e.backtestMode,"me-2":!e.backtestMode}])},[K(i,{id:"trade-filter",modelValue:v(t),"onUpdate:modelValue":s[0]||(s[0]=l=>tt(t)?t.value=l:null),type:"text",placeholder:"Filter",class:"w-full"},null,8,["modelValue"])],2),m("ul",Vt,[(c(!0),u(N,null,et(v(o),l=>(c(),u("li",{key:l.pair,button:"",class:P(["flex cursor-pointer last:rounded-b justify-between items-center px-1 py-1.5",{"bg-primary dark:border-primary text-primary-contrast":l.pair===v(n).activeBot.selectedPair}]),title:`${("formatPriceCurrency"in a?a.formatPriceCurrency:v(at))(l.profitAbs,v(n).activeBot.stakeCurrency,v(n).activeBot.stakeCurrencyDecimals)} - ${l.pair} - ${l.tradeCount} trades`,onClick:y=>v(n).activeBot.selectedPair=l.pair},[m("div",null,[rt(nt(l.pair)+" ",1),l.locks?(c(),u("span",{key:0,title:l.lockReason},[K(b)],8,zt)):$("",!0)]),l.trade&&!e.backtestMode?(c(),k(g,{key:0,trade:l.trade},null,8,["trade"])):$("",!0),e.backtestMode&&l.tradeCount>0?(c(),k(p,{key:1,"profit-ratio":l.profit,"stake-currency":v(n).activeBot.stakeCurrency},null,8,["profit-ratio","stake-currency"])):$("",!0)],10,It))),128))])])}}}),ae=st(Kt,[["__scopeId","data-v-a7bbb5d1"]]);var M={name:"ChevronLeftIcon",extends:it};function Ot(e,t,r,n,o,a){return c(),u("svg",d({width:"14",height:"14",viewBox:"0 0 14 14",fill:"none",xmlns:"http://www.w3.org/2000/svg"},e.pti()),t[0]||(t[0]=[m("path",{d:"M9.61296 13C9.50997 13.0005 9.40792 12.9804 9.3128 12.9409C9.21767 12.9014 9.13139 12.8433 9.05902 12.7701L3.83313 7.54416C3.68634 7.39718 3.60388 7.19795 3.60388 6.99022C3.60388 6.78249 3.68634 6.58325 3.83313 6.43628L9.05902 1.21039C9.20762 1.07192 9.40416 0.996539 9.60724 1.00012C9.81032 1.00371 10.0041 1.08597 10.1477 1.22959C10.2913 1.37322 10.3736 1.56698 10.3772 1.77005C10.3808 1.97313 10.3054 2.16968 10.1669 2.31827L5.49496 6.99022L10.1669 11.6622C10.3137 11.8091 10.3962 12.0084 10.3962 12.2161C10.3962 12.4238 10.3137 12.6231 10.1669 12.7701C10.0945 12.8433 10.0083 12.9014 9.91313 12.9409C9.81801 12.9804 9.71596 13.0005 9.61296 13Z",fill:"currentColor"},null,-1)]),16)}M.render=Ot;var Et={root:"p-tablist",content:function(t){var r=t.instance;return["p-tablist-content",{"p-tablist-viewport":r.$pcTabs.scrollable}]},tabList:"p-tablist-tab-list",activeBar:"p-tablist-active-bar",prevButton:"p-tablist-prev-button p-tablist-nav-button",nextButton:"p-tablist-next-button p-tablist-nav-button"},Rt=T.extend({name:"tablist",classes:Et}),Ft={name:"BaseTabList",extends:w,props:{},style:Rt,provide:function(){return{$pcTabList:this,$parentInstance:this}}},Dt={name:"TabList",extends:Ft,inheritAttrs:!1,inject:["$pcTabs"],data:function(){return{isPrevButtonEnabled:!1,isNextButtonEnabled:!0}},resizeObserver:void 0,watch:{showNavigators:function(t){t?this.bindResizeObserver():this.unbindResizeObserver()},activeValue:{flush:"post",handler:function(){this.updateInkBar()}}},mounted:function(){var t=this;setTimeout(function(){t.updateInkBar()},150),this.showNavigators&&(this.updateButtonState(),this.bindResizeObserver())},updated:function(){this.showNavigators&&this.updateButtonState()},beforeUnmount:function(){this.unbindResizeObserver()},methods:{onScroll:function(t){this.showNavigators&&this.updateButtonState(),t.preventDefault()},onPrevButtonClick:function(){var t=this.$refs.content,r=this.getVisibleButtonWidths(),n=S(t)-r,o=Math.abs(t.scrollLeft),a=n*.8,s=o-a,i=Math.max(s,0);t.scrollLeft=O(t)?-1*i:i},onNextButtonClick:function(){var t=this.$refs.content,r=this.getVisibleButtonWidths(),n=S(t)-r,o=Math.abs(t.scrollLeft),a=n*.8,s=o+a,i=t.scrollWidth-n,b=Math.min(s,i);t.scrollLeft=O(t)?-1*b:b},bindResizeObserver:function(){var t=this;this.resizeObserver=new ResizeObserver(function(){return t.updateButtonState()}),this.resizeObserver.observe(this.$refs.list)},unbindResizeObserver:function(){var t;(t=this.resizeObserver)===null||t===void 0||t.unobserve(this.$refs.list),this.resizeObserver=void 0},updateInkBar:function(){var t=this.$refs,r=t.content,n=t.inkbar,o=t.tabs;if(n){var a=V(r,'[data-pc-name="tab"][data-p-active="true"]');this.$pcTabs.isVertical()?(n.style.height=lt(a)+"px",n.style.top=B(a).top-B(o).top+"px"):(n.style.width=ct(a)+"px",n.style.left=B(a).left-B(o).left+"px")}},updateButtonState:function(){var t=this.$refs,r=t.list,n=t.content,o=n.scrollTop,a=n.scrollWidth,s=n.scrollHeight,i=n.offsetWidth,b=n.offsetHeight,g=Math.abs(n.scrollLeft),p=[S(n),ot(n)],l=p[0],y=p[1];this.$pcTabs.isVertical()?(this.isPrevButtonEnabled=o!==0,this.isNextButtonEnabled=r.offsetHeight>=b&&parseInt(o)!==s-y):(this.isPrevButtonEnabled=g!==0,this.isNextButtonEnabled=r.offsetWidth>=i&&parseInt(g)!==a-l)},getVisibleButtonWidths:function(){var t=this.$refs,r=t.prevButton,n=t.nextButton,o=0;return this.showNavigators&&(o=((r==null?void 0:r.offsetWidth)||0)+((n==null?void 0:n.offsetWidth)||0)),o}},computed:{templates:function(){return this.$pcTabs.$slots},activeValue:function(){return this.$pcTabs.d_value},showNavigators:function(){return this.$pcTabs.scrollable&&this.$pcTabs.showNavigators},prevButtonAriaLabel:function(){return this.$primevue.config.locale.aria?this.$primevue.config.locale.aria.previous:void 0},nextButtonAriaLabel:function(){return this.$primevue.config.locale.aria?this.$primevue.config.locale.aria.next:void 0},dataP:function(){return D({scrollable:this.$pcTabs.scrollable})}},components:{ChevronLeftIcon:M,ChevronRightIcon:pt},directives:{ripple:F}},Wt=["data-p"],Mt=["aria-label","tabindex"],Ht=["data-p"],jt=["aria-orientation"],Ut=["aria-label","tabindex"];function qt(e,t,r,n,o,a){var s=W("ripple");return c(),u("div",d({ref:"list",class:e.cx("root"),"data-p":a.dataP},e.ptmi("root")),[a.showNavigators&&o.isPrevButtonEnabled?C((c(),u("button",d({key:0,ref:"prevButton",type:"button",class:e.cx("prevButton"),"aria-label":a.prevButtonAriaLabel,tabindex:a.$pcTabs.tabindex,onClick:t[0]||(t[0]=function(){return a.onPrevButtonClick&&a.onPrevButtonClick.apply(a,arguments)})},e.ptm("prevButton"),{"data-pc-group-section":"navigator"}),[(c(),k(_(a.templates.previcon||"ChevronLeftIcon"),d({"aria-hidden":"true"},e.ptm("prevIcon")),null,16))],16,Mt)),[[s]]):$("",!0),m("div",d({ref:"content",class:e.cx("content"),onScroll:t[1]||(t[1]=function(){return a.onScroll&&a.onScroll.apply(a,arguments)}),"data-p":a.dataP},e.ptm("content")),[m("div",d({ref:"tabs",class:e.cx("tabList"),role:"tablist","aria-orientation":a.$pcTabs.orientation||"horizontal"},e.ptm("tabList")),[h(e.$slots,"default"),m("span",d({ref:"inkbar",class:e.cx("activeBar"),role:"presentation","aria-hidden":"true"},e.ptm("activeBar")),null,16)],16,jt)],16,Ht),a.showNavigators&&o.isNextButtonEnabled?C((c(),u("button",d({key:1,ref:"nextButton",type:"button",class:e.cx("nextButton"),"aria-label":a.nextButtonAriaLabel,tabindex:a.$pcTabs.tabindex,onClick:t[2]||(t[2]=function(){return a.onNextButtonClick&&a.onNextButtonClick.apply(a,arguments)})},e.ptm("nextButton"),{"data-pc-group-section":"navigator"}),[(c(),k(_(a.templates.nexticon||"ChevronRightIcon"),d({"aria-hidden":"true"},e.ptm("nextIcon")),null,16))],16,Ut)),[[s]]):$("",!0)],16,Wt)}Dt.render=qt;var Zt={root:function(t){var r=t.instance,n=t.props;return["p-tab",{"p-tab-active":r.active,"p-disabled":n.disabled}]}},Gt=T.extend({name:"tab",classes:Zt}),Xt={name:"BaseTab",extends:w,props:{value:{type:[String,Number],default:void 0},disabled:{type:Boolean,default:!1},as:{type:[String,Object],default:"BUTTON"},asChild:{type:Boolean,default:!1}},style:Gt,provide:function(){return{$pcTab:this,$parentInstance:this}}},Jt={name:"Tab",extends:Xt,inheritAttrs:!1,inject:["$pcTabs","$pcTabList"],methods:{onFocus:function(){this.$pcTabs.selectOnFocus&&this.changeActiveValue()},onClick:function(){this.changeActiveValue()},onKeydown:function(t){switch(t.code){case"ArrowRight":this.onArrowRightKey(t);break;case"ArrowLeft":this.onArrowLeftKey(t);break;case"Home":this.onHomeKey(t);break;case"End":this.onEndKey(t);break;case"PageDown":this.onPageDownKey(t);break;case"PageUp":this.onPageUpKey(t);break;case"Enter":case"NumpadEnter":case"Space":this.onEnterKey(t);break}},onArrowRightKey:function(t){var r=this.findNextTab(t.currentTarget);r?this.changeFocusedTab(t,r):this.onHomeKey(t),t.preventDefault()},onArrowLeftKey:function(t){var r=this.findPrevTab(t.currentTarget);r?this.changeFocusedTab(t,r):this.onEndKey(t),t.preventDefault()},onHomeKey:function(t){var r=this.findFirstTab();this.changeFocusedTab(t,r),t.preventDefault()},onEndKey:function(t){var r=this.findLastTab();this.changeFocusedTab(t,r),t.preventDefault()},onPageDownKey:function(t){this.scrollInView(this.findLastTab()),t.preventDefault()},onPageUpKey:function(t){this.scrollInView(this.findFirstTab()),t.preventDefault()},onEnterKey:function(t){this.changeActiveValue(),t.preventDefault()},findNextTab:function(t){var r=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!1,n=r?t:t.nextElementSibling;return n?x(n,"data-p-disabled")||x(n,"data-pc-section")==="activebar"?this.findNextTab(n):V(n,'[data-pc-name="tab"]'):null},findPrevTab:function(t){var r=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!1,n=r?t:t.previousElementSibling;return n?x(n,"data-p-disabled")||x(n,"data-pc-section")==="activebar"?this.findPrevTab(n):V(n,'[data-pc-name="tab"]'):null},findFirstTab:function(){return this.findNextTab(this.$pcTabList.$refs.tabs.firstElementChild,!0)},findLastTab:function(){return this.findPrevTab(this.$pcTabList.$refs.tabs.lastElementChild,!0)},changeActiveValue:function(){this.$pcTabs.updateValue(this.value)},changeFocusedTab:function(t,r){dt(r),this.scrollInView(r)},scrollInView:function(t){var r;t==null||(r=t.scrollIntoView)===null||r===void 0||r.call(t,{block:"nearest"})}},computed:{active:function(){var t;return E((t=this.$pcTabs)===null||t===void 0?void 0:t.d_value,this.value)},id:function(){var t;return"".concat((t=this.$pcTabs)===null||t===void 0?void 0:t.$id,"_tab_").concat(this.value)},ariaControls:function(){var t;return"".concat((t=this.$pcTabs)===null||t===void 0?void 0:t.$id,"_tabpanel_").concat(this.value)},attrs:function(){return d(this.asAttrs,this.a11yAttrs,this.ptmi("root",this.ptParams))},asAttrs:function(){return this.as==="BUTTON"?{type:"button",disabled:this.disabled}:void 0},a11yAttrs:function(){return{id:this.id,tabindex:this.active?this.$pcTabs.tabindex:-1,role:"tab","aria-selected":this.active,"aria-controls":this.ariaControls,"data-pc-name":"tab","data-p-disabled":this.disabled,"data-p-active":this.active,onFocus:this.onFocus,onKeydown:this.onKeydown}},ptParams:function(){return{context:{active:this.active}}},dataP:function(){return D({active:this.active})}},directives:{ripple:F}};function Qt(e,t,r,n,o,a){var s=W("ripple");return e.asChild?h(e.$slots,"default",{key:1,dataP:a.dataP,class:P(e.cx("root")),active:a.active,a11yAttrs:a.a11yAttrs,onClick:a.onClick}):C((c(),k(_(e.as),d({key:0,class:e.cx("root"),"data-p":a.dataP,onClick:a.onClick},a.attrs),{default:R(function(){return[h(e.$slots,"default")]}),_:3},16,["class","data-p","onClick"])),[[s]])}Jt.render=Qt;export{ae as _,Dt as a,Jt as b,wt as c,Pt as d,gt as s};
//# sourceMappingURL=index-Dg8pG0Nf.js.map
