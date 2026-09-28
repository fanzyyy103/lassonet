# Concrete: Original Endpoint-Wise BIC QLN

## PI Coverage and Mean Width
| Interval | Model | Target | Covered Count | Test N | PI Coverage | Coverage Error | Mean PI Length | Median PI Length | Std PI Length | Crossing Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 95% | QR | 0.95 | 190 | 206 | 0.92233 | 0.02767 | 36.536015 | 36.086489 | 7.019074 | 0.0 |
| 95% | QL | 0.95 | 191 | 206 | 0.927184 | 0.022816 | 36.748201 | 36.390011 | 7.066212 | 0.0 |
| 95% | QLN Endpoint-BIC | 0.95 | 193 | 206 | 0.936893 | 0.013107 | 43.297573 | 42.341632 | 7.366196 | 0.0 |
| 90% | QR | 0.9 | 176 | 206 | 0.854369 | 0.045631 | 31.128577 | 29.307657 | 10.598521 | 0.0 |
| 90% | QL | 0.9 | 176 | 206 | 0.854369 | 0.045631 | 31.340206 | 29.490044 | 10.739255 | 0.0 |
| 90% | QLN Endpoint-BIC | 0.9 | 184 | 206 | 0.893204 | 0.006796 | 35.683873 | 34.978672 | 7.577002 | 0.0 |

## Selected Parameters
| Model | Tau | Selected Parameter | Selected Count | Selected Features |
| --- | --- | --- | --- | --- |
| QR | 0.025 | 0.0 |  |  |
| QL | 0.025 | 0.001 | 7 | [Cement (component 1)(kg in a m^3 mixture), Blast Furnace Slag (component 2)(kg in a m^3 mixture), Fly Ash (component 3)(kg in a m^3 mixture), Water  (component 4)(kg in a m^3 mixture), Superplasticizer (component 5)(kg in a m^3 mixture), Coarse Aggregate  (component 6)(kg in a m^3 mixture), Age (day)] |
| QLN Endpoint-BIC | 0.025 | 70.144879 | 8 | [Cement (component 1)(kg in a m^3 mixture), Blast Furnace Slag (component 2)(kg in a m^3 mixture), Fly Ash (component 3)(kg in a m^3 mixture), Water  (component 4)(kg in a m^3 mixture), Superplasticizer (component 5)(kg in a m^3 mixture), Coarse Aggregate  (component 6)(kg in a m^3 mixture), Fine Aggregate (component 7)(kg in a m^3 mixture), Age (day)] |
| QR | 0.05 | 0.0 |  |  |
| QL | 0.05 | 0.001 | 7 | [Cement (component 1)(kg in a m^3 mixture), Blast Furnace Slag (component 2)(kg in a m^3 mixture), Fly Ash (component 3)(kg in a m^3 mixture), Water  (component 4)(kg in a m^3 mixture), Superplasticizer (component 5)(kg in a m^3 mixture), Fine Aggregate (component 7)(kg in a m^3 mixture), Age (day)] |
| QLN Endpoint-BIC | 0.05 | 85.261538 | 7 | [Cement (component 1)(kg in a m^3 mixture), Blast Furnace Slag (component 2)(kg in a m^3 mixture), Water  (component 4)(kg in a m^3 mixture), Superplasticizer (component 5)(kg in a m^3 mixture), Coarse Aggregate  (component 6)(kg in a m^3 mixture), Fine Aggregate (component 7)(kg in a m^3 mixture), Age (day)] |
| QR | 0.95 | 0.0 |  |  |
| QL | 0.95 | 1e-06 | 8 | [Cement (component 1)(kg in a m^3 mixture), Blast Furnace Slag (component 2)(kg in a m^3 mixture), Fly Ash (component 3)(kg in a m^3 mixture), Water  (component 4)(kg in a m^3 mixture), Superplasticizer (component 5)(kg in a m^3 mixture), Coarse Aggregate  (component 6)(kg in a m^3 mixture), Fine Aggregate (component 7)(kg in a m^3 mixture), Age (day)] |
| QLN Endpoint-BIC | 0.95 | 119.971547 | 8 | [Cement (component 1)(kg in a m^3 mixture), Blast Furnace Slag (component 2)(kg in a m^3 mixture), Fly Ash (component 3)(kg in a m^3 mixture), Water  (component 4)(kg in a m^3 mixture), Superplasticizer (component 5)(kg in a m^3 mixture), Coarse Aggregate  (component 6)(kg in a m^3 mixture), Fine Aggregate (component 7)(kg in a m^3 mixture), Age (day)] |
| QR | 0.975 | 0.0 |  |  |
| QL | 0.975 | 0.0003 | 8 | [Cement (component 1)(kg in a m^3 mixture), Blast Furnace Slag (component 2)(kg in a m^3 mixture), Fly Ash (component 3)(kg in a m^3 mixture), Water  (component 4)(kg in a m^3 mixture), Superplasticizer (component 5)(kg in a m^3 mixture), Coarse Aggregate  (component 6)(kg in a m^3 mixture), Fine Aggregate (component 7)(kg in a m^3 mixture), Age (day)] |
| QLN Endpoint-BIC | 0.975 | 119.971547 | 8 | [Cement (component 1)(kg in a m^3 mixture), Blast Furnace Slag (component 2)(kg in a m^3 mixture), Fly Ash (component 3)(kg in a m^3 mixture), Water  (component 4)(kg in a m^3 mixture), Superplasticizer (component 5)(kg in a m^3 mixture), Coarse Aggregate  (component 6)(kg in a m^3 mixture), Fine Aggregate (component 7)(kg in a m^3 mixture), Age (day)] |

## Plot
- /Users/zhongyangfan/Documents/lassonet 2/outputs/concrete_qr_ql_qln_endpoint_bic_90_95/concrete_qr_ql_qln_endpoint_bic_90_95.png