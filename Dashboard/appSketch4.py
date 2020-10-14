


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import cv2
import base64
import os
import json
import numpy as np
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring
import pandas as pd
from PIL import Image



######################################################################################################


### Needed Paths ###
path_to_images = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/'
path_to_mask = '/Users/natewagner/Documents/Mote_Manatee_Project/canny_filled2.png'
path_to_blank = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated.jpg'




def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            #dbc.NavItem(dbc.NavLink("Home", href="/index")),
            #dbc.NavItem(dbc.NavLink("Database", href="/index")),           
        ],
        brand="Manatee Identification",
        color="primary",
        expand = 'sm',
        dark=True,
    )
    return navbar


class Compare_ROIS(object):
    def __init__(self, path, input_sketch, roi, mask):
        self.path = path
        self.mask = mask
        self.input_sketch = input_sketch
        self.roi = roi #[x1,y1,x2,y2]
        self.processed_images = None
    def compare_rois(self):
        # get ROI array
        input_contour_info = []
        for input_bb in self.roi:  
            input_sketch_roi = self.input_sketch[int(input_bb[0]): int(input_bb[1]), int(input_bb[2]): int(input_bb[3])]
            # preprocess input sketch roi
            input_roi = self.preprocess(input_sketch_roi)        
            # find contours in input sketch roi
            input_contours = cv2.findContours(input_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # find contour rois in input sketch roi       
            input_shapes, input_area, input_num_contour, input_bb_dim = self.find_contours(input_contours[0], input_roi)
            input_contour_info.append([input_shapes, input_area, input_num_contour, input_bb_dim])
#            Image.fromarray(input_roi).show()

        distance_dict = []
    # First get all file names in list file_names
        for i in range(len(self.processed_images)):
            # get ROI array and preprocess     
            for x in range(len(self.roi)):
                sketch_roi = self.processed_images[i][1][int(self.roi[x][0]): int(self.roi[x][1]), int(self.roi[x][2]): int(self.roi[x][3])]
                # find contours in ROI
                contours = cv2.findContours(sketch_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # get contours rois in sketch roi                
                contours_shapes, contour_area, num_contours, bb_dims  = self.find_contours(contours[0], sketch_roi)  
                distances = self.compute_distances(input_contour_info[x][3], bb_dims, str(self.processed_images[i][0]))  
                if distances != "NA":
                    distance_dict.append((str(self.processed_images[i][0]), distances))                     
        distance_dict_df = pd.DataFrame(distance_dict)
        unique_names_cnts = distance_dict_df.groupby(0)[1].agg(['count', 'mean'])
        unique_names_cnts['names'] = unique_names_cnts.index
        has_all_scars = unique_names_cnts[unique_names_cnts['count'] >= len(self.roi)]
        returned = has_all_scars[['names', 'mean']]        
        returned_list = returned.values.tolist()     
        returned_list = sorted(returned_list, key = lambda x: x[1])         
        returned_list2 = []
        for idx, img in enumerate(returned_list):
            returned_list2.append([idx+1, img[0], img[1]])
        return returned_list2
    def preprocess(self, img):
        # blur
        img = cv2.blur(img, (2,2))
        # black background
        img = cv2.bitwise_not(img)
        # threshold
        _,img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)        
        return img
    def find_contours(self, contours_list, sketch_roi):
        contours_rois = []
        contour_area = []
        num_contours = 0
        bb_dims = []
        for contour in contours_list:
            #filled = cv2.fillPoly(sketch_roi.copy(), contours_list, 255)           
            x, y, w, h = cv2.boundingRect(contour)
            roi = sketch_roi[y:y + h, x:x + w]            
            # contour center coordinates
            contour_x = round(x + (w/2)) 
            contour_y = round(y + (h/2))               
            area = cv2.contourArea(contour)  
            # extent
            rect_area = w*h
            extent = float(area)/rect_area
            avg_pixel = np.sum(roi)
            if area > 20 and len(contour) >= 5:
                (x,y), (MA,ma), angle = cv2.fitEllipse(contour)   
                if MA == 0:
                    MA = 1
                aspect_ratio = float(ma)/MA                           
                contours_rois.append(roi)
                contour_area.append(area)
                bb_dims.append([np.array([w,h]), area, angle, np.array([MA,ma]), np.array([contour_x, contour_y]), avg_pixel, aspect_ratio, extent]) 
                num_contours += 1
        return contours_rois, contour_area, num_contours, bb_dims    
    def compute_distances(self, input_contours_shape, contours_shape, name):
        num_input_scars = len(input_contours_shape)
        num_scars = len(contours_shape)
        if num_input_scars == 0 and num_scars > 0:
            return 'NA'
        if num_input_scars == 0 and num_scars == 0:
            return 0
        #if num_input_scars != 0 and num_scars != 0:
        #if num_input_scars <= num_scars and num_scars < num_input_scars+3:
        if num_input_scars != 0 and num_scars != 0:
            comparisons = []
            for shape in input_contours_shape:                
                for num, shape2 in enumerate(contours_shape):
                    # Separate h,w and MA,ma and x,y
                    input_h, input_w = shape[0]
                    h,w = shape2[0]
                    input_MA, input_ma = shape[3]  
                    MA, ma = shape2[3]
                    input_x, input_y = shape[4]
                    x,y = shape2[4]    
                    input_area = shape[1]
                    area = shape2[1]
                    input_aspect = shape[6]
                    aspect = shape2[6]                    
                    # Compute percentage differences for each feature
                    diff_in_x = abs(input_x - x)
                    percentage_in_x = (100*diff_in_x)/input_x
                    diff_in_y = abs(input_y - y)
                    percentage_in_y = (100*diff_in_y)/input_y
                    diff_in_MA = abs(input_MA - MA)
                    percentage_MA = (100*diff_in_MA)/input_MA
                    diff_in_ma = abs(input_ma - ma)/ input_ma
                    percentage_ma = (100*diff_in_ma)/input_ma
                    diff_in_area = abs(input_area - area)/ input_area
                    percentage_area = (100*diff_in_area)/input_area
                    diff_in_aspect = abs(input_aspect - aspect)/ input_aspect
                    percentage_aspect = (100*diff_in_aspect)/input_aspect                    
                    #diff_in_pixs = abs(shape[5] - shape2[5])
                    #percentage_area = (100*(diff_in_pixs))/shape[5]
                    diff_in_angle = abs(shape[2] - shape2[2])
                    percentage_angle = (100*(diff_in_angle))/shape[2]
                    # if name == 'U2867_B.jpg':
                    #     print('U2867_B.jpg')
                    #     print((0.08 * percentage_angle, 0.23 * percentage_MA, 0.23 * percentage_ma,  0.10 * percentage_area, 0.36 * percentage_aspect))
                    #     print(1/5*(0.08 * percentage_angle + 0.23 * percentage_MA + 0.23 * percentage_ma + 0.10 * percentage_area + 0.36 * percentage_aspect))
                    # if name == 'U3163_A.jpg':
                    #     print('U3163_A.jpg')
                    #     print((0.08 * percentage_angle, 0.23 * percentage_MA, 0.23 * percentage_ma,  0.10 * percentage_area, 0.36 * percentage_aspect))
                    #     print(1/5*(0.08 * percentage_angle + 0.23 * percentage_MA + 0.23 * percentage_ma + 0.10 * percentage_area + 0.36 * percentage_aspect))
                    #comparisons.append(np.mean([percentage_area, percentage_angle, percentage_MA, percentage_ma, percentage_in_x, percentage_in_y]))
                    comparisons.append([num, 1/5*(0.08 * percentage_angle + 0.23 * percentage_MA + 0.23 * percentage_ma + 0.10 * percentage_area + 0.36 * percentage_aspect)])
            if len(comparisons) != 0:
                distances = self.computeScore(comparisons, num_input_scars)                                    
            return np.mean(distances)
        else:
            return 'NA'
    def removeOutline(self, img, mask):
        mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(mask,(5,5),0)
        mask = cv2.addWeighted(blur,1.5,mask,-0.5,0)
        mask[mask != 0] = 1          
        img = cv2.resize(img, (259, 559), interpolation= cv2.INTER_NEAREST)
        img[mask == 1] = 255
        return img
    def preLoadData(self):
        sketch_names = [] 
        processed_images = []
        for file_ in sorted(os.listdir(self.path)):
            if file_[-1] == 'g':
                sketch_names.append(str(file_))
            if file_[-1] == 'G':
                sketch_names.append(str(file_))
        for i in range(len(sketch_names)):
            # get sketch path
            sketch_path = str(self.path + str(sketch_names[i]))
            # read sketch in grayscale format
            sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
            sketch_no_outline = self.removeOutline(sketch, self.mask)
            preprocessed_img = self.preprocess(sketch_no_outline)
            processed_images.append([str(sketch_names[i]), preprocessed_img])
        self.processed_images = processed_images
    def computeScore(self, dist, num_input_scars):        
        scores = []
        num_lookup_scars = len(list(set([el[0] for el in dist]))) 
        while len(scores) <= num_input_scars - 1:
            if len(scores) >= num_lookup_scars:
                break
            current_lowest = dist[np.argmin([el[1] for el in dist])]        
            if len(dist) != 0:
                scores.append(current_lowest)
            dist = [item for item in dist if item[0] != current_lowest[0]]    
        return np.sum([el[1] for el in scores])


# initiate class:
find_matches_func = Compare_ROIS(path_to_images, None, None, path_to_mask)
find_matches_func.preLoadData()


######################################################################################################


# LITERA
app = dash.Dash(__name__, meta_tags=[{"content": "width=device-width"}], external_stylesheets=[dbc.themes.LITERA])




# dash canvas info
filename = Image.open(path_to_blank)
canvas_width = 259   # this has to be set to 259 because we use the canvas as input to the model



score_html = " "
n_html = " "
num_matches_html = " "
name_html = " "
name_info = None

app.layout = html.Div(
    [
     Navbar(),
     html.Br(),
     dbc.Row(
         [
             dbc.Col(
                 dbc.Card(
                     [
                     dbc.CardHeader(
                     html.H1("Sketch", className="card-text"), style={"width": "100%", 'textAlign': 'center'}
                     ),
                         dbc.Row([
                                dbc.Col(
                                    html.Div(                                       
                                            DashCanvas(
                                                id='canvas',
                                                width=canvas_width,
                                                filename=filename,
                                                hide_buttons=['line', 'select', 'zoom', 'pan'],
                                                goButtonTitle="Search"
                                                ), className="canvas-outer",
                                        ),
                                    style = {'padding-left': '100px'},
                                    md = 8),
                                 dbc.Col(
                                             dbc.Card(
                                                 [
                    
                                                    # daq.BooleanSwitch(
                                                    #         id='my-boolean-switch',
                                                    #         label="Orientation",
                                                    #         on=True
                                                    #     ),
                                                    # daq.BooleanSwitch(
                                                    #         id='my-boolean-switch1',
                                                    #         label="Width",                                            
                                                    #         on=True
                                                    #     ),
                                                    # daq.BooleanSwitch(
                                                    #         id='my-boolean-switch2',
                                                    #         label="Height",                                            
                                                    #         on=True
                                                    #     ),                                
                                                    html.Div(
                                                        [
                                                        html.H6(children=['Brush Width'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                                                            dcc.Slider(
                                                                id='bg-width-slider',
                                                                min=1,
                                                                max=40,
                                                                step=1,
                                                                value=1,
                                                            ),                                         
                                                        html.H6(children=['Brush Color'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                                                            dcc.RadioItems(
                                                                    id='color-picker',
                                                                    options=[
                                                                        {'label': 'Black', 'value': dict(hex='#000000')},
                                                                        {'label': 'Grey', 'value': dict(hex='#666666')},
                                                                    ],
                                                                    value=dict(hex='#000000'),
                                                                    labelStyle={'display': 'inline-block', 'margin-right': '20px', 'margin-left': '20px', 'font-weight': 300},
                                                                    inputStyle={"margin-right": "10px"},
                                                                    style={'textAlign': 'center', 'font-weight': 'normal', 'font-size' : '15'}
                                                                ),
                                                    html.Div(
                                                        html.Button(id="clear", children="clear", style = {'width': '76%', 'margin-left': '17px', 'margin-right': '10px', 'margin-bottom': '5px'}),
                                                        className="v-card-content-markdown-outer",
                                                        )
                                                            ],
                                                            style = {"margin-top": "10px"}, className="three columns"),                                                   
                                                    ],
                                                 style = {'width': '100%',
                                                          'margin-top': '100px',
                                                          'margin-right': '100px',
                                                          'display': 'inline-block',                                      
                                                          'box-shadow': '4px 4px 4px grey'}
                                                     ), 
                                             style = {'padding-right': '130px'},
                                             md = 4),
                                 ])
                    ],
                    style = {'align-items': 'center',
                             'width': '100%',
                             #'margin-left': '40px',
                             'display': 'inline-block',
                             'box-shadow': '8px 8px 8px grey'}
                             #'margin-bottom': '10px'}
                    ),
                    md = 6),               
             dbc.Col(
                 dbc.Card(
                     [
                         dbc.CardHeader(
                         html.H1("Browse Matches", className="card-text", style = {'textAlign': 'center'}), style={"width": "100%"}
                         ),
                         html.Div(
                             [
                                 dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button("Left", id="left_click", color="primary", className="mr-2", n_clicks = 0, size="lg"),
                                                    style = {'margin-top': '20px',
                                                             'vertical-align': 'middle'},
                                                md = 2),
                                            dbc.Col(
                                                html.Span(id="sketch_output", style={"vertical-align": "middle"}
                                                ),
                                                md = 8),
                                            dbc.Col(
                                                dbc.Button("Right", id="right_click", color="primary", className="mr-2", n_clicks = 0, size="lg"),
                                                    style = {'margin-top': '20px',
                                                             'vertical-align': 'middle'},
                                                md = 2),
                                        ]
                                    ),
                             ],
                             style = {'textAlign': 'center',
                                      'width': '100%',
                                      'align-items': 'center',
                                      }),
                          dbc.CardFooter(
                              dbc.Row(
                                  [
                                      dbc.Col(html.Div(id = 'sketch_output_info1'), md = 4),
                                      dbc.Col(html.Div(id = 'sketch_output_info2'), md = 4),
                                      dbc.Col(html.Div(id = 'sketch_output_info3'), md = 4)
                                  ]
                                  )
                              )
                     ],
                     style={'width': '100%',
                            'align-items': 'center',
                            'display': 'inline-block',
                            'box-shadow': '8px 8px 8px grey'}
                     ),
                     md = 6),
        ]
        ),
    ],
    style={'padding': '0px 40px 0px 40px'}
)





t = None

@app.callback(Output('sketch_output', 'data'),
                [Input('canvas', 'json_data')],
                [State('canvas', 'image_content')])
def update_data(string, image):    
    global name_info, names, switch, count, find_matches_func, num_returned, t    
    switch = True
    is_rect = False
    if string:
        data = json.loads(string)
        bb_info = data['objects'][1:]         
        bounding_box_list = []        
        for i in bb_info:
            if i['type'] == 'rect':  
                is_rect = True
                top = i['top']
                left = i['left']
                wd = i['width']
                ht = i['height']
                bounding_box_list.append((top, top+ht, left, left+wd))
            else:
                continue        
        if is_rect == False:
            bounding_box_list.append((0, 559, 0, 259))        
        mask = parse_jsonstring(string, shape=(559, 259))
        mask = (~mask.astype(bool)).astype(int)
        mask[mask == 1] = 255
        mask = mask.astype(np.uint8)
        find_matches_func.input_sketch = mask
        find_matches_func.roi = bounding_box_list
        matches = find_matches_func.compare_rois()
        name_info = matches
        is_rect = False     
    return






last_right = False
blank = base64.b64encode(open(path_to_blank, 'rb').read())

def return_image(n):
    global path_to_images
    global names
    global num_returned
    global count
    global last_right
    global score_html, n_html, num_matches_html, name_info, name_html        
    if name_info is not None:
        if n <= 0:
            file = name_info[n][1]
        else:
            file = next(i[1] for i in name_info if i[0] == n) 
        encoded_image = base64.b64encode(open(path_to_images + file, 'rb').read())
        if name_info is None:
            return html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
            ], style = {'align-items': 'center'})
        else:
            score_html = str(round(next(i[2] for i in name_info if i[0] == n), 2))
            name_html = str(file)[0:-4]
            n_html = str(next(i[0] for i in name_info if i[0] == n))
            num_matches_html = str(len(name_info))
            return html.Div([
                #html.H5(str(file)[0:-4]),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
                ], style = {'align-items': 'center'})





count = 0
switch = False

@app.callback(
    [Output("sketch_output", "children"),
     Output("sketch_output_info1", "children"),
     Output("sketch_output_info2", "children"),
     Output("sketch_output_info3", "children")],
    [
     Input("right_click", "n_clicks"),
     Input("left_click", "n_clicks"),
     Input('canvas', 'json_data')
     ]
)
def on_button_click(n, n2, run):
    global count
    global switch
    global names
    global score_html, n_html, num_matches_html, name_html    
    if switch == True:
        return_image(1)        
        count = 0
        switch = False   
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if name_info is None:
        return html.Img(src='data:image/png;base64,{}'.format(blank.decode())), html.H4('Score:    ' + str(score_html), style = {'width':'100%', 'textAlign': 'left'}), html.H4(" ", style = {'width':'100%', 'textAlign': 'center'}),  html.H4('Matches:    ' + str(n_html) + '/' + str(num_matches_html), style = {'textAlign': 'right'})
    if "right_click" in changed_id:
        if abs(count) == len(name_info):
            count = 0
        count += 1
        return return_image(count), html.H4('Score:    ' + str(score_html), style = {'width':'100%', 'textAlign': 'left'}), html.H4(name_html, style = {'width':'100%', 'textAlign': 'center'}), html.H4('Matches:    ' + str(n_html) + '/' + str(num_matches_html), style = {'textAlign': 'right'})
    if "left_click" in changed_id:
        if count >= 2:
            count -= 1        
            return return_image(count), html.H4('Score:    ' + str(score_html), style = {'width':'100%', 'textAlign': 'left'}), html.H4(name_html, style = {'width':'100%', 'textAlign': 'center'}), html.H4('Matches:    ' + str(n_html) + '/' + str(num_matches_html), style = {'textAlign': 'right'})
        else:
            return return_image(count), html.H4('Score:    ' + str(score_html), style = {'width':'100%', 'textAlign': 'left'}), html.H4(name_html, style = {'width':'100%', 'textAlign': 'center'}), html.H4('Matches:    ' + str(n_html) + '/' + str(num_matches_html), style = {'textAlign': 'right'})


    
@app.callback(Output("canvas", "json_objects"), [Input("clear", "n_clicks")])
def clear_canvas(n):
    if n is None:
        return dash.no_update
    strings = ['{"objects":[ ]}', '{"objects":[]}']
    return strings[n % 2]



@app.callback(Output('canvas', 'lineColor'),
            [Input('color-picker', 'value')])
def update_canvas_linecolor(value):
    if isinstance(value, dict):
        return value['hex']
    else:
        return value


@app.callback(Output('canvas', 'lineWidth'),
            [Input('bg-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value






if __name__ == '__main__':
    app.run_server()
