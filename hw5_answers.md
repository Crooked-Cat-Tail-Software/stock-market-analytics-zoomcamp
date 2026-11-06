# Homework 5: Three Stock Research Ideas (Draft)

Assignment: [homework5.md](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework5.md)



| | Idea 1: Imaging AI | Idea 2: Indian tractors | Idea 3: Robot vacuums |
|---|---|---|---|
| Mechanism | Payment and regulatory events | Rainfall and dealer inventory | Consumer sales share |
| Market | US and Europe | India | China-listed, global sales |
| Main data | FDA and CMS records | Rainfall and monthly industry sales | Marketplace ranks, reviews and prices |
| Horizon | Days to 12 months | 1 to 3 months | About one quarter |

---

## Question 1: In imaging AI, does getting paid matter more than getting cleared?

**1. Market and scope.** US-listed and European medical imaging companies: GE HealthCare (GEHC), Siemens Healthineers (SHL.DE), Philips (PHIA.AS), RadNet (RDNT) and HeartFlow (HTFL). The horizon has two parts: a few trading days around each event, and 6 to 12 months for any revenue effect.

**2. Investment thesis.** FDA clearance has become common, so it may no longer tell investors much. Payment is much scarcer. Reporting on the FDA list counted 1,524 AI devices through March 2026, about 76% of them in radiology, while only two or three AI tools were reported to have permanent CPT billing codes. I expect a new payment route to change a product's revenue more than one more clearance does, because a hospital or imaging center has a clearer reason to use a tool it can bill for. The example I would start with is CPT code 75577 for AI coronary plaque analysis, which took effect on January 1, 2026 at a national average of about $1,021 in imaging centers and physician offices and $951 in hospital outpatient settings.

**3. Potential edge.** "AI device approved" is an easy headline, so investors may overreact to clearances and pay less attention to payment rules, which are long and technical. Reading CMS fee schedules and matching them to specific products takes industry knowledge. The edge is not that this information is hidden. It is that the rules are slow to read and hard to connect to individual companies.

**4. Data and sources.**
- *FDA AI-enabled device list:* shows decision date, company, device, lead panel and product code, and downloads as CSV, Excel or XML. It measures how many clearances each company receives. **Checked directly.** The latest entry I saw was dated June 29, 2026.
- *openFDA 510(k) API:* gives device-level decision dates and applicants. **Checked directly** with a test query, which returned decisions from September 2026. Applicant names vary (for example, "GE Healthcare Finland Oy"), so they must be mapped to parent companies.
- *Payment details for 75577:* **checked through trade press and a company release**, not in the CMS final rule itself.
- *HeartFlow results:* full-year 2026 plaque revenue guidance rose from $19-21M after Q1 to $29-31M after Q2, about 12% of total guidance at the midpoint. **Q1 came from a press release. Q2 came from a secondary summary and should be confirmed.**
- *Price history:* **checked directly** on Yahoo Finance. HeartFlow has only about 16 months of monthly prices and GE HealthCare begins in December 2022.
- *Still to verify:* the CMS rule text and publication dates, how many AI payment events exist, and when each clearance first became public.

**5. Research approach.** I would run an event study of abnormal returns against a healthcare benchmark. Group A is FDA clearances, and group B is CMS proposed and final payment decisions that affect a named product. The first comparison is the return over the five trading days after each event. The second is whether group B companies show faster revenue growth over the next 6 to 12 months. An event study fits because the thesis is about which type of news carries information. The decision it informs is whether to hold a payment-exposed company instead of a broad imaging basket after a payment event.

The event date needs care. The FDA decision date is not always when the market found out, so I would use the first public date I can establish, such as a company press release. The FDA list also updates only periodically, and the March 2026 data was reported as released in mid-June, so using list data before its release date would be look-ahead bias.

**6. Validation and risks.** The benchmark is an equal-weight basket of the imaging stocks, plus a healthcare ETF such as XLV. I would reject the idea if payment events show no larger abnormal return than clearances, or if any move happens entirely on the announcement day and cannot be captured after costs. The biggest limit is sample size. There may be fewer than ten usable payment events, so this could only be an exploratory case study. AI is also a small part of revenue at the three large companies, so it may not show up in their results. HeartFlow's short history and its growth for other reasons could also be mistaken for a payment effect. Payment can also be cut in later years.

---

## Question 2: Do rainfall and dealer inventory predict Indian tractor makers' next results?

**1. Market and scope.** India's National Stock Exchange, focusing on Mahindra & Mahindra (M&M.NS) and Escorts Kubota (ESCORTS.NS). The horizon is 1 to 3 months. This is my route into small-farm technology: tractors are the mechanization purchase for Indian farmers, and farm income there depends heavily on rainfall.

**2. Investment thesis.** Two things should move tractor demand before it reaches company results. First, monsoon rain affects farm income and so the demand for tractors. Second, manufacturers record a sale when a tractor ships to a dealer, not when a farmer buys it. If shipments keep running ahead of retail sales, dealers are building stock, and future shipments may need to fall. A stock pile-up after a weak monsoon is the case I would expect to be most damaging.

**3. Potential edge.** Headline monthly sales are widely reported, so I do not expect an edge from those alone. Any edge would come from combining sources that are usually read separately: rainfall by region rather than the national average, and retail data alongside wholesale data. A company with a strong headline number could still be weakening if its dealers are building inventory. Understanding India's crop seasons and regional demand also matters here.

**4. Data and sources.**
- *FADA retail tractor data:* monthly dealer-to-customer sales based on vehicle registration records, with company market shares. **Checked through trade press.** Coverage can vary. For example, September 2026 figures reportedly exclude Telangana.
- *TMA wholesale data:* monthly manufacturer shipments. **Checked through trade press only.** I have not found the full official series.
- *IMD rainfall data:* the 2026 monsoon reportedly ended at about 87% of the long-period average, the lowest since 2015. **Checked through news reports.** I have not found the downloadable regional data, and the baseline differs slightly between sources.
- *Recent sales:* retail rose 27.8% in July 2026 and 0.9% in August (trade press). This makes late 2026 a useful live test of the rainfall idea, although I cannot yet attribute that slowdown to the monsoon.
- *Price history:* **checked directly** on Yahoo Finance, with 10 years of monthly prices for both stocks.
- *Still to verify:* publication dates for each dataset, FADA's treatment of commercial-use tractors, and how to weight rainfall by each region's tractor demand.

**5. Research approach.** I would build two monthly signals. The first is rainfall as a percentage of normal, weighted by each region's share of tractor sales. The second is the change in the wholesale-minus-retail gap over three months, for each company where the data allows it. I would use changes rather than levels because the two datasets are not directly comparable. In February 2026, FADA reported 89,418 retail units and TMA reported 78,901 domestic units, even though wholesale is usually expected to be higher.

I would regress the next quarter's tractor volume growth and the next 1-to-3-month stock returns on these signals, using only data available on each decision date. A simple regression fits because there are few observations and a complex model would overfit. The decision it informs is whether to reduce or hold exposure to a tractor maker before its next results.

**6. Validation and risks.** The benchmarks are the Nifty 50, the Nifty Auto index, and a naive forecast that next quarter's growth equals last quarter's. I would reject the idea if the signals do not beat the naive forecast, if the stock has already moved by the time the data is published, or if the inventory gap fails to predict shipment declines.

The main risks are that ten years contains only about ten monsoons, that monthly sales are public quickly and may be priced in, and that festival timing distorts month-to-month comparisons. M&M also has a large automotive business, which dilutes the tractor signal, so Escorts Kubota may be the cleaner test. Government support, interest rates and crop prices can all override rainfall. Direct trading in Indian stocks may also be restricted depending on where an investor lives, so this may stay a paper test at first. Transaction costs and liquidity would still need checking.

---

## Question 3: Do marketplace signals reveal robot vacuum share shifts before quarterly results?

**1. Market and scope.** Shanghai-listed Roborock (688169.SS) and Ecovacs (603486.SS). The horizon is about one quarter, from the signal to the next earnings report. I chose robot vacuums rather than humanoid robots because vacuums are the household robots with real sales to measure today.

**2. Investment thesis.** This market changes share quickly because buyers compare products and prices online. Its leader is not safe either: iRobot, with a reported 42% US share, filed for Chapter 11 in December 2025, and shareholders were not expected to receive equity in the reorganized company. I expect rank and review changes on major marketplaces to show which brand is winning before quarterly reports confirm it. I also expect the quality of that growth to matter, because share gained through heavy discounting may hurt margins. So the thesis is not "robots are growing." It is that the winner within the category is visible early in public consumer data.

**3. Potential edge.** Analysts covering Chinese companies may focus on domestic channel data, even though Roborock's overseas sales were reported to have passed its domestic sales in 2025. Combining rank, reviews and discount depth across US and European marketplaces could show changes that are not yet in reported results. Knowing how a shopper actually compares these products also helps in deciding which models and features matter.

**4. Data and sources.**
- *Company reports:* revenue, margin and overseas sales from Shanghai Stock Exchange filings, which are the outcome to predict. **Not yet checked at the source.** Roborock's first-half 2026 revenue of RMB 10.08 billion comes from secondary reporting.
- *Amazon rank and price history:* for example, from Keepa, which would measure relative demand and discounting. **To verify,** including its cost, its coverage and how far back it goes.
- *Review counts:* a rough measure of units sold, using review growth rather than total reviews. **To verify.** Collection must follow each site's terms.
- *Google Trends by brand and country:* a free measure of consumer interest. **To verify.** Values are relative and can change between downloads.
- *China channel data:* retail tracking firms such as AVC. **To verify.** It is probably paid.
- *Market share:* a commercial report puts Roborock at 23% of global sales in 2025. **Checked, but its method could not be verified,** so I would use it only as context.
- *Price history:* **checked directly** on Yahoo Finance, from February 2020 for Roborock and May 2018 for Ecovacs.
- *CSI 300 index:* **checked directly, and Yahoo Finance returned only one month of history,** so I would need another source for this benchmark.
- *Hong Kong listing:* Roborock's application reportedly lapsed in June 2026. **One secondary source only.** I would use the Shanghai shares for now.

**5. Research approach.** For the top-selling models of each brand, I would build a weekly demand score from rank and review growth, weighted by price, and a discount score measuring how far prices sit below their recent normal level. For each quarter, I would then ask two questions: does the change in Roborock's demand share relative to Ecovacs predict which company reports stronger sequential revenue growth, and does a high discount score predict weaker gross margin?

If both hold, I would test the return over the 20 trading days after results, then whether the signal works before the report. I would start with a simple rank comparison rather than a machine learning model, because there are only about 25 quarters and two companies. The decision it informs is which of the two stocks to prefer before earnings. Short selling A-shares may not be practical, so I would treat this as a relative preference rather than a true long-short trade.

**6. Validation and risks.** The benchmarks are an equal-weight mix of the two stocks, the CSI 300 index, and a naive rule of preferring whichever company grew faster last quarter. I would reject the idea if the demand score does not beat that naive rule, or if the link disappears once price and seasonality are accounted for.

The biggest practical risk is data availability. If dated history is too short or expensive, the test cannot be done properly. Amazon is only one channel, so a good signal there may miss China, Europe or offline sales. Prime Day and Singles' Day will dominate the seasonal pattern. Rank can also be bought through advertising, and reviews can be manipulated or merged across models. Quarterly reports are published with a delay, and A-shares have daily price limits and possible access limits for foreign investors. Other strong brands, such as Dreame and Xiaomi, could take share from both companies, so a gain against one rival may not mean a gain in the whole market. Using only surviving companies would also add bias, so iRobot should be kept in any wider study.

---

## Sources

- [FDA AI-enabled device list](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices)
- [openFDA 510(k) API](https://open.fda.gov/apis/device/510k/)
- [Radiology Business: 68 new FDA-cleared algorithms](https://radiologybusiness.com/topics/artificial-intelligence/radiology-gets-68-new-fda-cleared-algorithms)
- [Radiology Business: Medicare imaging AI payment](https://radiologybusiness.com/topics/artificial-intelligence/medicare-physician-fee-schedule-includes-cpt-code-1000-payment-amount-imaging-ai-software)
- [Elucid: CMS reimbursement for plaque analysis](https://www.businesswire.com/news/home/20251203262228/en/Elucid-Gains-Broad-CMS-Reimbursement-of-Coronary-Plaque-Analysis-Using-PlaqueIQ)
- [HeartFlow Q1 2026 results](https://www.biospace.com/press-releases/heartflow-reports-first-quarter-2026-financial-results-and-raises-full-year-2026-guidance)
- [HeartFlow Q2 2026 summary](https://quartr.com/events/heartflow-inc-htfl-q2-2026_FVw2SKlR)
- [Autocar Pro: February 2026 TMA data](https://www.autocarpro.in/news/domestic-tractor-shipments-grow-34-in-february-exports-up-8-131656)
- [Khetigaadi: February 2026 retail sales](https://khetigaadi.com/news/?p=9509)
- [Tractorgyan: July 2026 retail sales](https://tractorgyan.com/tractor-industry-news-blogs/2747/retail-tractor-sales-in-july-2026)
- [Tractorgyan: August 2026 retail sales](https://tractorgyan.com/tractor-industry-news-blogs/2857/retail-tractor-sales-in-august-2026)
- [Autocar Pro: September 2026 retail coverage](https://www.autocarpro.in/news/tractor-and-ce-retail-show-mixed-september-performance-fada-135072)
- [2026 monsoon deficit summary](https://superkalam.com/current-affairs/01-10-2026/southwest-monsoon-ends-with-12-deficit-october-rain-also-likely-below-normal-imd-pg1-c0741b13-efac-431c-8afe-e6fcd2df8f02)
- [Fortune: iRobot bankruptcy](https://fortune.com/2025/12/16/roomba-parent-company-irobot-files-for-bankruptcy/?rand=8593)
- [Research and Markets: robot vacuum market](https://www.researchandmarkets.com/reports/6235817/robot-vacuum-cleaner-market-consolidation)
- [EqualOcean: Roborock overseas sales and Hong Kong plan](https://equalocean.com/news/2025060921564-roborock-announces-planned-hong-kong-listing-overseas-sales-surpass-domestic)
- [Revenue Memo: Roborock ownership and Hong Kong application](https://www.revenuememo.com/p/who-owns-roborock)
