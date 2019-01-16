/////////////////////////////////////////////////////////////////////////////
//                                                                         //
//     OBJECTS DETECTION AND TRACKING VIA PERCEPTUAL RELEVANCE METRICS     //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////



///  FILE INCLUDES  ///
#include "./PRMovement/pr_movement.h"
#include "./PRMovement/bmpreader.h"
#include "./PRMovement/yuv_rgb.h"
#include "./PRMovement/imgutils.h"
#include "./PRMovement/perceptual_relevance_api.h"
#include <math.h>



///  PREPROCESSOR MACROS  ///
#define BUFF_SIZE_OBJECT_DETECTION 2
#define MAX_NUM_RECTANGLES 5

#define MIN_PR_DIFF_TO_CONSIDER_CUMULUS 0.25    //0.25
#define THRESHOLD_KEEP_RECTANGLE_EDGE 0.25      //0.3

#define TOP_EDGE 1
#define LEFT_EDGE 2
#define RIGHT_EDGE 3
#define BOTTOM_EDGE 4

#define POSSIBLE_CUMULUS 1
#define NOT_A_CUMULUS 0

#define MAX_CUMULUS_SIZE 8

#define MODE_BASE_IMAGE_FIRST_FRAME 1
#define MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME 2

#define MAX_MOV_VALID_FRAME_MODE_FIRST_FRAME 0.065
#define MAX_MOV_VALID_FRAME_MODE_PREV_FRAME 0.035
#define MOV_INCR_PER_FRAME 0.01
#define VALID_FRAME 1
#define INVALID_FRAME 0

#define MAX_RECTANGLE_CHANGE_PER_FRAME 1
#define MAX_FRAMES_NOT_SEEN_UNTIL_REMOVING_RECTANGLE 10 //5 //15 //120

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define TRACE_LEVEL 0   // How much information is printed in the console. The higher, the more info is printed
                        // -1 shows nothing, 0 shows basic, etc. Accepted values: -1, 0, 1, 2, 3



///  GLOBAL VARIABLES  ///

/**
 * Rectangle: to represent the object locations (2 ints define left upper corner and 2 other ints, the right lower corner)
 *  x1, y1
 *  +------------------+
 *  |                  |
 *  |                  |
 *  |                  |
 *  +------------------+ x2, y2
 */
typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
} Rectangle;

/**
 * Cumulus: to represent the zones where there are many blocks with "reatively high" pr difference respect to the ref frame
 *   +---------------+  -+           -+
 *   |               |   |            |
 *   |               |   |- size      |
 *   |     (x, y)    |   |            |
 *   |       +       |  _|            |- 2*size - 1
 *   |               |                |
 *   |               |                |
 *   |               |                |
 *   +---------------+               -+
 */
typedef struct {
    int x_center;
    int y_center;
    int cumulus_size;
} Cumulus;

// The number of rectangles
int num_rects;


// The mode used to detect objects (reference frame is always the first o the inmediately previous to the current)
int mode;

// Simple linked list (with head and tail pointers) for saving the rectangles
typedef struct linkedrectangle{
    Rectangle *data;
    int frames_not_seen;
    struct linkedrectangle *next;
} LinkedRectangle;
LinkedRectangle *rect_list_head;
LinkedRectangle *rect_list_tail;

// A mask indicating if the object detector should find rectangles in some blocks or not (rectangles overlapping only
//when moving, not in finding)
int **mask;



///  FUNCTION PROTOTYPES  ///
void rectangles_free();
LinkedRectangle* rectangle_list_add(Rectangle **rect);
int rectangle_list_remove(LinkedRectangle **lr);

void init_mask();
void clean_mask();
int add_rectangles_to_mask();
void add_rectangle_to_mask(LinkedRectangle **lr);

void pr_changes(int image_position_buffer);

int find_objects();

int track_objects();
LinkedRectangle* track_object(LinkedRectangle **lr);

int is_cumulus_seed(int block_x, int block_y);

Cumulus get_cumulus_centered(int block_x, int block_y);
float* cumulus_pr_neighbours(int block_x, int block_y, int cumulus_size);
float sum_pr_diffs(int x_center, int y_center, int cumulus_size);
Rectangle* cumulus_to_rectangle(Cumulus cumulus);

void reduce_rectangle_size(Rectangle *rect);
int drop_upper_rows(Rectangle rect);
int drop_lower_rows(Rectangle rect);
int drop_left_columns(Rectangle rect);
int drop_right_columns(Rectangle rect);

int drawEdgeOfRectangle(int block_x, int block_y, int whichEdge);
void draw_rectangles_in_frame();

void update_reference_frame(int position);
int is_frame_valid (int position);



///  FUNCTION IMPLEMENTATIONS  ///

// SUBSTITUTE with an adapted pr_to_movement to have position buffer tunable to use the differences with the frame 0
void pr_changes(int image_position_buffer) {
    /*int ref_image_position_buffer;
    
    switch (mode) {
        case MODE_BASE_IMAGE_FIRST_FRAME:
            if (image_position_buffer==0) {
                diffs_x = pr_x_buff[0];
                diffs_y = pr_y_buff[0];
                return;
            }
            ref_image_position_buffer = 0;
            break;
        case MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME:
            ref_image_position_buffer = image_position_buffer - 1;
            if (ref_image_position_buffer < 0)
                ref_image_position_buffer = buff_size - 1;
            break;
        default:
            printf("ERROR: mode not defined\n");
            return;
    }

    
    for (int i = 0; i < total_blocks_width + 1; i++){
        for (int j = 0; j < total_blocks_height + 1; j++){
            diffs_x[j][i] = pr_x_buff[image_position_buffer][j][i] - pr_x_buff[ref_image_position_buffer][j][i];
            diffs_y[j][i] = pr_y_buff[image_position_buffer][j][i] - pr_y_buff[ref_image_position_buffer][j][i];
            if (diffs_x[j][i] < 0) diffs_x[j][i] = -diffs_x[j][i];
            if (diffs_y[j][i] < 0) diffs_y[j][i] = -diffs_y[j][i];
        }
    }*/
    


    /*for (int i = 0; i < total_blocks_width + 1; i++){
        for (int j = 0; j < total_blocks_height + 1; j++){
            if (image_position_buffer==0) {
                diffs_x[j][i] = pr_x_buff[0][j][i];
                diffs_y[j][i] = pr_y_buff[0][j][i];
            } else {
                diffs_x[j][i] = pr_x_buff[image_position_buffer][j][i] - pr_x_buff[0][j][i];
                diffs_y[j][i] = pr_y_buff[image_position_buffer][j][i] - pr_y_buff[0][j][i];
            }
            if (diffs_x[j][i] < 0) diffs_x[j][i] = -diffs_x[j][i];
            if (diffs_y[j][i] < 0) diffs_y[j][i] = -diffs_y[j][i];
        }
    }*/

    int prev_image_position_buffer;
    switch (mode) {
        case MODE_BASE_IMAGE_FIRST_FRAME:
            for (int i = 0; i < total_blocks_width + 1; i++){
                for (int j = 0; j < total_blocks_height + 1; j++){
                    if (image_position_buffer==0) {
                        diffs_x[j][i] = pr_x_buff[0][j][i];
                        diffs_y[j][i] = pr_y_buff[0][j][i];
                    } else {
                        diffs_x[j][i] = pr_x_buff[image_position_buffer][j][i] - pr_x_buff[0][j][i];
                        diffs_y[j][i] = pr_y_buff[image_position_buffer][j][i] - pr_y_buff[0][j][i];
                    }
                    if (diffs_x[j][i] < 0) diffs_x[j][i] = -diffs_x[j][i];
                    if (diffs_y[j][i] < 0) diffs_y[j][i] = -diffs_y[j][i];
                }
            }
            break;


        case MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME:
            prev_image_position_buffer = image_position_buffer - 1;
            if (prev_image_position_buffer < 0)
                prev_image_position_buffer = buff_size - 1;

            for (int i = 0; i < total_blocks_width + 1; i++){
                for (int j = 0; j < total_blocks_height + 1; j++){
                    diffs_x[j][i] = pr_x_buff[image_position_buffer][j][i] - pr_x_buff[prev_image_position_buffer][j][i];
                    diffs_y[j][i] = pr_y_buff[image_position_buffer][j][i] - pr_y_buff[prev_image_position_buffer][j][i];
                    if (diffs_x[j][i] < 0) diffs_x[j][i] = -diffs_x[j][i];
                    if (diffs_y[j][i] < 0) diffs_y[j][i] = -diffs_y[j][i];
                }
            }
            break;
        default:
            printf("ERROR: mode not defined\n");
            return;
    }
}

/**
 * Allocates memory for the mask. And calls clean_mask() to initialize the values.
 */
void init_mask(){
    mask = malloc(total_blocks_width * sizeof(int *));
    for (int y=0; y < total_blocks_width; ++y) {
        mask[y] = malloc(total_blocks_height * sizeof(int));
    }

    clean_mask();
}

/**
 * Sets the mask to 1 in the borders (the two outer "blocks" in the frame) and 0 in the rest.
 */
void clean_mask(){
    for (int y = 0; y < total_blocks_height; ++y) {
        for (int x = 0; x < total_blocks_width; ++x) {
            if (y<2 || y>total_blocks_height-2 || x<2 || x>total_blocks_width-2)
                mask[x][y] = 1;
            else
                mask[x][y] = 0;
        }
    }
}

/**
 * Adds the areas occupied by the stored rectangles to the mask, so that next rectangles found are not overlapping with
 * current ones.
 *
 * @return int
 *      Returns 0 if everything was OK, other values if something went wrong. Note: -1 if list is corrupt.
 */
int add_rectangles_to_mask(){
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0) return 0; // all null (list empty)
    else if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0) return -1; // at least one null (list corrupt)
    else{
        LinkedRectangle *p = rect_list_head;
        LinkedRectangle *p_before;
        do {
            add_rectangle_to_mask(&p);
            p_before = p;
            p = p->next;
        } while (p_before!=rect_list_tail);
    }
    return 0;
}

/**
 * Adds the area occupied by the rectangle to the mask, so that next rectangles found are not overlapping with it.
 *
 * @param **LinkedRectangle
 */
void add_rectangle_to_mask(LinkedRectangle **lr){
    int x, y;
    Rectangle *rect = (*lr)->data;

    for (x = rect->x1; x <= rect->x2; ++x) {
        for (y = rect->y1; y <= rect->y2; ++y) {
            mask[x][y] = 1;
        }
    }
}

/**
 * Frees the resorces allocated for storing the rectangles.
 */
void rectangles_free() {
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0) return; // all null (list empty)
    else if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0) printf("ERROR list corrupt.\n");  // at least one null (list corrupt)
    else{
        LinkedRectangle *p = rect_list_head;
        LinkedRectangle *p_before;
        do {
            p_before = p;
            p = p->next;
            rectangle_list_remove(&p_before);
        } while (p!=NULL);
    }
    if (!(rect_list_head==NULL && rect_list_tail==NULL && num_rects==0)) printf("ERROR freeing rectangles list.\n");
}

/**
 * Adds the specified rectangle to the end of the linked list. Updates num_rects(+=1), *rect_list_head and *rect_list_tail and
 * allocates memory space for the node.
 *
 * @param Rectangle **rect
 *      The pointer to the rectangle to be added to the list. Note: rectangle memory must be already allocated.
 * 
 * @return LinkedRectangle*
 *      The rectangle node created.
 */
LinkedRectangle* rectangle_list_add(Rectangle **rect) {
    if (num_rects>=MAX_NUM_RECTANGLES){                                         // list is full
        free(*rect);
        (*rect) = NULL;
        return NULL;
    }

    LinkedRectangle *lr = (LinkedRectangle *)malloc(sizeof(LinkedRectangle));
    if (lr==NULL) {                                                             // no memory
        free(*rect);
        (*rect) = NULL;
        return NULL;
    }
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0)           // all null (list empty)
        rect_list_head = lr; // the new rectangle is the first element of the list
    else if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0)      // one null but the other(s) not
        return NULL;    // nonsense, both first and last must be null or not
    else                                                                        // all not null (typical non-empty list)
        rect_list_tail->next = lr;

    rect_list_tail = lr; // the new rectangle is the last element of the list
    lr->data = *rect;
    lr->frames_not_seen = 0;
    lr->next = NULL;
    num_rects++;
    if (TRACE_LEVEL>=1) printf("Rectangulo guardado:\tx1=%d\ty1=%d\tx2=%d\ty2=%d\n", (lr->data)->x1, (lr->data)->y1, (lr->data)->x2, (lr->data)->y2);
    return lr;
}

/**
 * Removes a specified rectangle from the linked list.  Updates num_rects(-=1), *rect_list_head and *rect_list_tail and
 * frees memory space of the node.
 *
 * @param LinkedRectangle **lr
 *      The rectangle to remove from the list.
 * 
 * @return int
 *      Returns 0 if node was removed correctly, 1 in other cases (null reference, empty list, ect.)
 */
int rectangle_list_remove(LinkedRectangle **lr) {
    if (*lr==NULL) return 1;                           // bad parameter
    if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0) return 1;      // any null (list empty or corrupt)
    
    if (*lr==rect_list_head) {
        if (*lr==rect_list_tail) {                      // is head and tail
            rect_list_tail = NULL;
            rect_list_head = NULL;
        }else{                                          // is only head
            rect_list_head = rect_list_head->next;
        }
        free((*lr)->data);
        (*lr)->data = NULL;
        free(*lr);
        *lr = NULL;
        num_rects--;
        return 0;
    }

    LinkedRectangle *p = rect_list_head;
    LinkedRectangle *p_before = rect_list_head;
    while (p_before!=rect_list_tail) {
        p = p->next;
        if (p==*lr) {
            if (*lr==rect_list_tail) {                  // is only tail
                rect_list_tail = p_before;
                p_before->next = NULL;
            }else {                                     // is typical node
                p_before->next = p->next;
            }
            free((*lr)->data);
            (*lr)->data = NULL;
            free(*lr);
            *lr = NULL;
            num_rects--;
            return 0;
        }
        p_before=p;
    }
}

/**
 * Computes the sum of the pr differences (with the reference image) of all the blocks in the cumulus. If the given center
 * is out of the borders of the image, a value of 0.0 is returned, in order not to be selected as the best center.
 *
 * @param int x_center
 *      The x coordinate of the block which is the center of the cumulus.    
 * @param int y_center
 *      The y coordinate of the block which is the center of the cumulus.
 * @param int cumulus_size
 *      Indicates the size of the cumulus. 1 means 1x1 cumulus, 2 means 2x2, etc.
 *
 * @return float
 *      The sum of the pr differences (with the reference image) of all the blocks in the cumulus.
 */
float sum_pr_diffs(int x_center, int y_center, int cumulus_size) {
    if ((x_center<0 || y_center<0 || x_center>total_blocks_width || y_center>total_blocks_height)) {
        return 0.0;
    }

    float sum;
    int x_min, y_min, x_max, y_max;
    int a, b, i;

    i = cumulus_size;
    a = (i-(1+((i-1)%2)))/2;
    b = (i-(i%2))/2;

    x_min = x_center - a;
    y_min = y_center - a;
    x_max = x_center + b;
    y_max = y_center + b;

    //x_min = x_center - cumulus_size/2;
    //y_min = y_center - cumulus_size/2;
    //x_max = x_center + cumulus_size/2 + cumulus_size%2;
    //y_max = y_center + cumulus_size/2 + cumulus_size%2;

    sum = 0.0;
    for (int x = x_min; x <= x_max; ++x) {   //if ..x<=x_max.. then get_block_movement will segfault (it avegrages with x+1)
        for (int y = y_min; y <= y_max; ++y) {   //same with y
            if (!(x<0 || y<0 || x>total_blocks_width-1 || y>total_blocks_height-1)){       // if NOT out of image bounds
                sum += get_block_movement(x, y);
            }
        }
    }
    return sum;
}

/**
 * Calculates the pr sum of all the blocks on the cumuli (given block coordinates and cumulus size) and the possible new centers (8 closest neighbours)
 * 
 * @param int block_x
 *      The x block coordinate.
 * @param int block_y
 *      The y block coordinate.
 * @param int cumulus_size
 *      Indicates the size of the cumulus. 1 means 1x1 cumulus, 2 means 2x2, etc.
 *
 * @return float*
 *      Returns the array of sums of PRs. Always 9 positions. Remember to free memory.
 */
float* cumulus_pr_neighbours(int block_x, int block_y, int cumulus_size) {
    int ij_index;
    float* pr_values = malloc(9 * sizeof(float));
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
           ij_index = (j+1) + 3*(i+1);
           pr_values[ij_index] = sum_pr_diffs(block_x+j, block_y+i, cumulus_size);
        }
    }
    return pr_values;
}

/**
 * Checks if a block and its closest environment is relevant enough to be a cumulus.
 *
 * @param int block_x
 *      The x coordinate of the block that is asked to be considered cumulus.    
 * @param int block_y
 *      The y coordinate of the block that is asked to be considered cumulus.
 *
 * @return int
 *      Returns NOT_A_CUMULUS is the coordinates do not meet the requirements to be consedered a cumulus, and POSSIBLE_CUMULUS if they are met.
 */
int is_cumulus_seed(int block_x, int block_y) {
    float *pr_values;
    int is_cumulus = NOT_A_CUMULUS;

    if (sum_pr_diffs(block_x, block_y, 1) >= MIN_PR_DIFF_TO_CONSIDER_CUMULUS) {
        pr_values = cumulus_pr_neighbours(block_x, block_y, 1);

        if (!(pr_values[0]==0.0 && pr_values[1]==0.0 && pr_values[2]==0.0 && \
              pr_values[3]==0.0 &&                      pr_values[5]==0.0 && \
              pr_values[6]==0.0 && pr_values[7]==0.0 && pr_values[8]==0.0)) {
            is_cumulus = POSSIBLE_CUMULUS;
        }
        free(pr_values);
        pr_values = NULL;
    }
    return is_cumulus;
}

/**
 * Given a block inside a cumulus, finds the Cumulus with the best-fit block as center.
 *
 * @param int block_x
 *      The x coordinate of the starting center.    
 * @param int block_y
 *      The y coordinate of the starting center.
 *
 * @return Cumulus
 *      The cumulus with the selected center and size.
 */
Cumulus get_cumulus_centered(int block_x, int block_y) {
    int cumulus_size, index_new_center;
    float *pr_values;
    Cumulus cumulus;
    cumulus.x_center = block_x;
    cumulus.y_center = block_y;

    for (cumulus_size = 3; cumulus_size <= MAX_CUMULUS_SIZE; ++cumulus_size) {
        do{
            index_new_center = 4;   // The current center. If we dont move, we will stop iterating
            pr_values = cumulus_pr_neighbours(cumulus.x_center, cumulus.y_center, cumulus_size);
            for (int i = 0; i < 9; ++i) {
                if (pr_values[index_new_center] < pr_values[i]) {
                    index_new_center = i;
                }
                if (TRACE_LEVEL>=3) printf("pr_values[%d] = %f\n", i, pr_values[i]);
            }
            // Update center coordinates
            if (index_new_center != 4) {
                cumulus.x_center += (index_new_center%3)-1;     //x_center += j
                cumulus.y_center += (index_new_center/3)-1;     //y_center += i
            }
            free(pr_values);
            pr_values = NULL;
        } while (index_new_center!=4);
        cumulus.cumulus_size = cumulus_size;
    }
    
    return cumulus;
}

/**
 * Transforms a cumulus to a rectangle. Allocates memory for the rectangle.
 *
 * @param Cumulus cumulus
 *      The cumulus to transform.    
 *
 * @return *Rectangle
 *      The rectangle that encloses the cumulus. Remember to free memory.
 */
Rectangle* cumulus_to_rectangle(Cumulus cumulus){
    Rectangle *rect = (Rectangle *)malloc(sizeof(Rectangle));
    if (rect==NULL){
        printf("ERROR: no memory\n");
        return NULL;
    }

    int a, b, i;

    i = cumulus.cumulus_size;
    a = (i-(1+((i-1)%2)))/2;
    b = (i-(i%2))/2;

    rect->x1 = cumulus.x_center - a;
    rect->y1 = cumulus.y_center - a;
    rect->x2 = cumulus.x_center + b;
    rect->y2 = cumulus.y_center + b;

    return rect;
}

/**
 * These four functions keep the same structure. They find the number of border lines that can be eliminated from one side (top/bottom/left/right)
 * of the rectangle to keep it the smallest but still emcompassing the object.
 *
 * @param Rectangle rect
 *      The rectangle to work with.    
 *
 * @return int
 *      The number of lines to remove (0 to keep it the same).
 */
int drop_upper_rows(Rectangle rect) {
    float max_pr_in_the_line;
    int deleted_rows = 0;
    int x1 = rect.x1;
    int y1 = rect.y1;
    int x2 = rect.x2;
    int y2 = rect.y2; 

    for (int i = y1; i <= y2; ++i) {
        max_pr_in_the_line = 0.0;
        for (int j = x1; j <=x2; ++j) {
            max_pr_in_the_line = MAX(max_pr_in_the_line, sum_pr_diffs(j, i, 1));
        }
        if (max_pr_in_the_line < THRESHOLD_KEEP_RECTANGLE_EDGE) {
            deleted_rows++;
        } else {
            return deleted_rows;
        }
    }
    return deleted_rows;
}
int drop_lower_rows(Rectangle rect) {
    float max_pr_in_the_line;
    int deleted_rows = 0;
    int x1 = rect.x1;
    int y1 = rect.y1;
    int x2 = rect.x2;
    int y2 = rect.y2;

    for (int i = y2; i >= y1; --i) {
        max_pr_in_the_line = 0.0;
        for (int j = x1; j <= x2; ++j) {
            max_pr_in_the_line = MAX(max_pr_in_the_line, sum_pr_diffs(j, i, 1));
        }
        if (max_pr_in_the_line < THRESHOLD_KEEP_RECTANGLE_EDGE) {
            deleted_rows++;
        } else {
            return deleted_rows;
        }
    }
    return deleted_rows;
}
int drop_left_columns(Rectangle rect) {
    float max_pr_in_the_line;
    int deleted_cols = 0;
    int x1 = rect.x1;
    int y1 = rect.y1;
    int x2 = rect.x2;
    int y2 = rect.y2;

    for (int j = x1; j <= x2; ++j) {
        max_pr_in_the_line = 0.0;
        for (int i = y1; i <= y2; ++i) {
            max_pr_in_the_line = MAX(max_pr_in_the_line, sum_pr_diffs(j, i, 1));
        }
        if (max_pr_in_the_line < THRESHOLD_KEEP_RECTANGLE_EDGE) {
            deleted_cols++;
        } else {
            return deleted_cols;
        }
    }
    return deleted_cols;
}
int drop_right_columns(Rectangle rect) {
    float max_pr_in_the_line;
    int deleted_cols = 0;
    int x1 = rect.x1;
    int y1 = rect.y1;
    int x2 = rect.x2;
    int y2 = rect.y2;

    for (int j = x2; j >= x1; --j) {
        max_pr_in_the_line = 0.0;
        for (int i = y1; i <= y2; ++i) {
            max_pr_in_the_line = MAX(max_pr_in_the_line, sum_pr_diffs(j, i, 1));
        }
        if (max_pr_in_the_line < THRESHOLD_KEEP_RECTANGLE_EDGE) {
            deleted_cols++;
        } else {
            return deleted_cols;
        }
    }
    return deleted_cols;
}

/**
 * Given a rectangle, keeps or reduces its size to convert it in the smallest rectangle possible that emcompasses the object.
 *
 * @param *Rectnagle rect
 *      The rectangle to work with.    
 */
void reduce_rectangle_size(Rectangle *rect) {
    (rect->y1) += drop_upper_rows(*rect);
    (rect->y2) -= drop_lower_rows(*rect);
    (rect->x1) += drop_left_columns(*rect);
    (rect->x2) -= drop_right_columns(*rect);
}

/**
 * Finds the objects in the frame. Starts analyzing the possible cumuli of blocks with high pr difference relative to the base image. After an
 *      iterative process of finding the best cumuli center and refining the size, defines the rectangle that encloses the object and saves it.
 *
 * @return int
 *      Irrelevant/not used.
 */
int find_objects() {
    Rectangle *rect;
    Cumulus cumulus;
    if (TRACE_LEVEL>=1) printf("finding...\n");
    for (int block_y=0; block_y<total_blocks_height; block_y++) {
        for (int block_x=0; block_x<total_blocks_width; block_x++) {
            if (TRACE_LEVEL>=3) printf("block_x=%d - - - block_y=%d\n", block_x, block_y);

            if (is_cumulus_seed(block_x, block_y)){
                if (TRACE_LEVEL>=3) printf("cumulus seed found. block_x = %d \tblock_y = %d\n", block_x, block_y);
                cumulus = get_cumulus_centered(block_x, block_y);

                rect = cumulus_to_rectangle(cumulus);
                reduce_rectangle_size(rect);

                rectangle_list_add(&rect);
                return 0; //just need one rectangle for this version
            }
        }
    }
}

/**
 * Tracks the objects in the previous frame, updating ther position and sizemage.
 *
 * @return int
 *      Returns 0 if everything was OK, other values if something went wrong. Note: -1 if list is corrupt.
 */
int track_objects(){
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0) return 0; // all null (list empty)
    else if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0) return -1; // at least one null (list corrupt)
    else{
        LinkedRectangle *p = rect_list_head;
        LinkedRectangle *p_before;
        do {
            p_before = p;
            p = track_object(&p);
            if (p == NULL) return 0;            // si se borra el último de la lista
        } while (p_before!=rect_list_tail);
    }
    return 0;
}

/**
 * Updates the rectangle position and size. There is a max size grow and a max movement.
 *
 * @param LinkedRectangle **lr
 *      The rectangle to be tracked (keep, update position and size, or delete).
 *
 * @return int
 *      0 if rectangle is updated or kept and 1 if rectangle is removed.
 * 
 * @return LinkedRectangle *next
 *      The next rectangle in the list.
 */
LinkedRectangle* track_object(LinkedRectangle **lr){
    int initial_x1, initial_x2, initial_y1, initial_y2;
    int upper_rows, lower_rows, left_columns, right_columns;
    int x1_enlarged, x2_enlarged, y1_enlarged, y2_enlarged, x1_enlargement, x2_enlargement, y1_enlargement, y2_enlargement;

    if ((*lr)==NULL) return NULL;

    Rectangle *rect = (*lr)->data;
    LinkedRectangle *next = (*lr)->next;
    int frames_not_seen = (*lr)->frames_not_seen;

    // Save initial values
    initial_x1 = rect->x1;
    initial_x2 = rect->x2;
    initial_y1 = rect->y1;
    initial_y2 = rect->y2;

    // Enlage rectangle
    //rect->x1 = MAX(rect->x1 - MAX_RECTANGLE_CHANGE_PER_FRAME, 0);
    //rect->x2 = MIN(rect->x2 + MAX_RECTANGLE_CHANGE_PER_FRAME, total_blocks_width);
    //rect->y1 = MAX(rect->y1 - MAX_RECTANGLE_CHANGE_PER_FRAME, 0);
    //rect->y2 = MIN(rect->y2 + MAX_RECTANGLE_CHANGE_PER_FRAME, total_blocks_height);

    x1_enlarged = MAX(rect->x1 - MAX_RECTANGLE_CHANGE_PER_FRAME, 0);
    x1_enlargement = rect->x1 - x1_enlarged;
    rect->x1 = x1_enlarged;

    x2_enlarged = MIN(rect->x2 + MAX_RECTANGLE_CHANGE_PER_FRAME, total_blocks_width-1);
    x2_enlargement = rect->x2 - x2_enlarged;
    rect->x2 = x2_enlarged;

    y1_enlarged = MAX(rect->y1 - MAX_RECTANGLE_CHANGE_PER_FRAME, 0);
    y1_enlargement = rect->y1 - y1_enlarged;
    rect->y1 = y1_enlarged;

    y2_enlarged = MIN(rect->y2 + MAX_RECTANGLE_CHANGE_PER_FRAME, total_blocks_height-1);
    y2_enlargement = rect->y2 - y2_enlarged;
    rect->y2 = y2_enlarged;


    upper_rows    = drop_upper_rows(*rect);
    rect->y1 += MIN(upper_rows, 2*MAX_RECTANGLE_CHANGE_PER_FRAME);

    lower_rows    = drop_lower_rows(*rect);
    rect->y2 -= MIN(lower_rows, 2*MAX_RECTANGLE_CHANGE_PER_FRAME);

    left_columns  = drop_left_columns(*rect);
    rect->x1 += MIN(left_columns, 2*MAX_RECTANGLE_CHANGE_PER_FRAME);

    right_columns = drop_right_columns(*rect);
    rect->x2 -= MIN(right_columns, 2*MAX_RECTANGLE_CHANGE_PER_FRAME);
    
    if (TRACE_LEVEL>=0) printf("\n---RECTANGLE---\n");
    if (TRACE_LEVEL>=1) printf("INICIAL  -->\tx1=%d\t y1=%d\t x2=%d\t y2=%d\n", initial_x1, initial_y1, initial_x2, initial_y2);
    if (TRACE_LEVEL>=1) printf("ENLARGED -->\tx1=%d\t y1=%d\t x2=%d\t y2=%d\n", x1_enlarged, y1_enlarged, x2_enlarged, y2_enlarged);
    if (TRACE_LEVEL>=1) printf("FINAL    -->\t");
    if (TRACE_LEVEL>=0)               printf("x1=%d\t y1=%d\t x2=%d\t y2=%d\n", rect->x1, rect->y1, rect->x2, rect->y2);
    if (TRACE_LEVEL>=0) printf("frames_not_seen = %d\n", frames_not_seen);

    //if (upper_rows + lower_rows >= 1 + initial_y2 - initial_y1  ||  left_columns + right_columns >= 1 + initial_x2 - initial_x1) {//(*rect).y2 - (*rect).y1 <= 0 || (*rect).x2 - (*rect).x1 <=0) {
    if (upper_rows + lower_rows >= 1 + y2_enlarged - y1_enlarged  ||  left_columns + right_columns >= 1 + x2_enlarged - x1_enlarged) {//(*rect).y2 - (*rect).y1 <= 0 || (*rect).x2 - (*rect).x1 <=0) {
        rect->y1 = initial_y1;
        rect->y2 = initial_y2;
        rect->x1 = initial_x1;
        rect->x2 = initial_x2;
        frames_not_seen++;
        if (frames_not_seen >= MAX_FRAMES_NOT_SEEN_UNTIL_REMOVING_RECTANGLE) {
            rectangle_list_remove(lr);
            if (TRACE_LEVEL>=1) printf("Rectangle removed\n");
            *lr = NULL;
            return next;
        }
    } else{
        frames_not_seen = 0;
    }
    (*lr)->frames_not_seen = frames_not_seen;
    
    return next;
}

/**
 * Draws the specified edge in the block specified changing the luminance and chrominances in the corresponding edge of the block.
 *
 * @param int block_x
 *      The x coordinate of the block in which the rectangle edge should be drawn.    
 * @param int block_y
 *      The y coordinate of the block in which the rectangle edge should be drawn.
 * @param int whichEdge
 *      Indicates which of the edges should be drawn in the block (top, bottom, left or right). Implemented with preprocesor definitions.
 *
 * @return int
 *      If everything is fine returns 0, if there was a problem, -1.
 */
int drawEdgeOfRectangle(int block_x, int block_y, int whichEdge) {
    int xini = block_x*theoretical_block_width;
    int xfin = xini+theoretical_block_width;
    int yini = block_y*theoretical_block_height;
    int yfin = yini+theoretical_block_height;

    if (xfin > width-theoretical_block_width)
        xfin = width;
    if (yfin > height-theoretical_block_height)
        yfin = height;

    switch (whichEdge) {
        case TOP_EDGE:
            for (int xx = xini; xx < xfin; xx++) {
                y[yini*width+xx] = 82;
                y[(yini+1)*width+xx] = 82;
            }
            for (int xx = xini/2; xx < xfin/2; xx++) {
                u[yini/2*width+xx] = 90;
                v[yini/2*width+xx] = 240;
            }
        break;

        case LEFT_EDGE:
            for (int yy = yini; yy < yfin; yy++) {
                y[yy*width+xini] = 82;
                y[yy*width+(xini+1)] = 82;
            }
            for (int yy = yini/2; yy < yfin/2; yy++) {
                u[yy*width+xini/2] = 90;
                v[yy*width+xini/2] = 240;
            }
        break;

        case RIGHT_EDGE:
            for (int yy = yini; yy < yfin; yy++) {
                y[yy*width+(xfin-1)] = 82;
                y[yy*width+(xfin-2)] = 82;
            }
            for (int yy = yini/2; yy < yfin/2; yy++) {
                u[yy*width+(xfin/2-1)] = 90;
                v[yy*width+(xfin/2-1)] = 240;
            }       
        break;

        case BOTTOM_EDGE:
            for (int xx = xini; xx < xfin; xx++) {
                y[(yfin-1)*width+xx] = 82;
                y[(yfin-2)*width+xx] = 82;
            }
            for (int xx = xini/2; xx < xfin/2; xx++) {
                u[(yfin/2-1)*width+xx] = 90;
                v[(yfin/2-1)*width+xx] = 240;
            }
        break;

        default:
            printf("ERROR\n");
            return -1;
    }
    return 0;
}

/**
 * Draws the rectangles in the frame changing the luminance and chrominances in the edges of the rectangles.
 */
void draw_rectangles_in_frame() {
    Rectangle *rect;
    LinkedRectangle *p = rect_list_head;
    // for each rectangle
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0) return; // all null (list empty)
    else while (p!=NULL){
        rect = p->data;
        for (int block_y = rect->y1; block_y <= rect->y2; block_y++) {
            for (int block_x = rect->x1; block_x <= rect->x2; block_x++) {
                // Draw top edges
                if (block_y==rect->y1){
                    drawEdgeOfRectangle(block_x, block_y, TOP_EDGE);
                    if (TRACE_LEVEL>=2) printf("TOP_EDGE\tblock_x=%d\tblock_y=%d\n", block_x, block_y);
                }
                // Draw left edge
                if (block_x==rect->x1){
                    drawEdgeOfRectangle(block_x, block_y, LEFT_EDGE);
                    if (TRACE_LEVEL>=2) printf("LEFT_EDGE\tblock_x=%d\tblock_y=%d\n", block_x, block_y);
                }
                // Draw right edge
                if (block_x==rect->x2){
                    drawEdgeOfRectangle(block_x, block_y, RIGHT_EDGE);
                    if (TRACE_LEVEL>=2) printf("RIGHT_EDGE\tblock_x=%d\tblock_y=%d\n", block_x, block_y);
                }
                // Draw bottom edge
                if (block_y==rect->y2){
                    drawEdgeOfRectangle(block_x, block_y, BOTTOM_EDGE);
                    if (TRACE_LEVEL>=2) printf("BOTTOM_EDGE\tblock_x=%d\tblock_y=%d\n", block_x, block_y);
                }
            }
        }
        p = p->next;
    }
}

/**
 * Updates the frame used as reference to the current frame.
 *
 * @param int position
 *      The current buff position.
 */
void update_reference_frame(int position){
    //pr_x_buff[0] = pr_x_buff[position];
    //pr_y_buff[0] = pr_y_buff[position];
    for (int block_y=0; block_y<total_blocks_height+1; block_y++) {
        for (int block_x=0; block_x<total_blocks_width+1; block_x++) {
            pr_x_buff[0][block_y][block_x] = pr_x_buff[position][block_y][block_x];
            pr_y_buff[0][block_y][block_x] = pr_y_buff[position][block_y][block_x];
        }
    }
}

/**
 * Verifies if a frame is valid or not.
 *
 * @param int block_x
 *      The x coordinate of the block in which the rectangle edge should be drawn.    
 * @param int block_y
 *      The y coordinate of the block in which the rectangle edge should be drawn.
 * @param int whichEdge
 *      Indicates which of the edges should be drawn in the block (top, bottom, left or right). Implemented with preprocesor definitions.
 *
 * @return int
 *      If frame is valid returns VALID_FRAME (1), else returns INVALID_FRAME (0).
 */
int is_frame_valid (int position){
    int is_valid = INVALID_FRAME;
    static int in_a_row_invalid_frames = 0; //first time is 0, then saves value between calls
    static float last_movement = 0.0; //first time is 0.0, then saves value between calls
    float movement = get_image_movement(0);
    //if (movement==0.0) return INVALID_FRAME;

    switch (mode){
        case MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME:
            if (movement < last_movement+MOV_INCR_PER_FRAME || movement <= MAX_MOV_VALID_FRAME_MODE_PREV_FRAME) {
                is_valid = VALID_FRAME;
                last_movement = movement;
                if (TRACE_LEVEL>=0) printf("Movement = %f\n", movement);
            }else{
                if (TRACE_LEVEL>=0) printf("! ! ! --> INVALID FRAME, Movement = %f\n", movement);
            }
            break;

        case MODE_BASE_IMAGE_FIRST_FRAME:
            if (movement <= MAX_MOV_VALID_FRAME_MODE_FIRST_FRAME) {
                is_valid = VALID_FRAME;
                in_a_row_invalid_frames = 0;
                if (TRACE_LEVEL>=0) printf("Movement = %f\n", movement);
            } else {
                if (TRACE_LEVEL>=0) printf("! ! ! --> INVALID FRAME \tMovement = %f\n", movement);
                in_a_row_invalid_frames++;
                if (in_a_row_invalid_frames >= 2) {
                    update_reference_frame(position);
                }
            }
            break;

        default: 
            printf("ERROR\n");
            return INVALID_FRAME;
    }

    return is_valid;
}



///  MAIN  ///
int main( int argc, char** argv ) {
    mode = MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME; //MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME   MODE_BASE_IMAGE_FIRST_FRAME
    
    if (TRACE_LEVEL>=0) printf("\n\
         ___________________________________________________________________________ \n\
        |                                                                           |\n\
        |      OBJECTS DETECTION AND TRACKING VIA PERCEPTUAL RELEVANCE METRICS      |\n\
        |___________________________________________________________________________|\n");

    char* imageName = argv[1];
    char* imageExt = argv[2];
    char* frameArg = argv[3];
    int frameNumber;
    char image[100];
    int frame = atoi(frameArg);
    initiated = 0;

    int starting_frame;

    if (mode==MODE_BASE_IMAGE_FIRST_FRAME){
        if (TRACE_LEVEL>=0) printf("\nUsing first frame as reference (auto-modified when 2 invalid frames in a row).\n");
        starting_frame = 0;
        buff_size = 2;
    }
    else if (mode == MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME){
        if (TRACE_LEVEL>=0) printf("\nUsing previous frame as reference.\n");
        starting_frame = 1;
        buff_size = BUFF_SIZE_OBJECT_DETECTION;
    }

    // Loop to process all the frames required
    for (frameNumber = starting_frame; frameNumber < frame; frameNumber++){

        if (frameArg == NULL){
            frameNumber = 0;
            strcpy(image, imageName);
            strcat(image, imageExt);
        } else {
            sprintf(frameArg,"%i",frameNumber);
            strcpy(image, imageName);
            strcat(image, frameArg);
            strcat(image, imageExt);
        }

        if (TRACE_LEVEL>=0) printf("\n\n\nInput: %s\n", image);

        BITMAPINFOHEADER bitmapInfoHeader;

        LoadBitmapFileProperties(image, &bitmapInfoHeader);
        width = bitmapInfoHeader.biWidth;
        height = bitmapInfoHeader.biHeight;
        rgb_channels = bitmapInfoHeader.biBitCount/8;
        
        if (TRACE_LEVEL>=3) printf("width = %d, height = %d, rgb_channels = %d\n", width, height, rgb_channels);
        if (initiated == 0) {
            init_pr_computation(width, height, rgb_channels);
            init_mask();
            initiated = 1;
        }
        
        rgb = load_frame(image, width, height, rgb_channels);

        const size_t y_stride = width + (16-width%16)%16;
        const size_t uv_stride = y_stride;
        const size_t rgb_stride = width*3 +(16-(3*width)%16)%16;

        rgb24_yuv420_std(width, height, rgb, rgb_stride, y, u, v, y_stride, uv_stride, YCBCR_601);

        free(rgb);
        rgb = NULL;
        
        int position; //int position = MIN(frameNumber, 1); //int position = (frameNumber-1)%BUFF_SIZE_OBJECT_DETECTION;
        if (mode==MODE_BASE_IMAGE_FIRST_FRAME)
            position = MIN(frameNumber, 1);
        else if (mode==MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME)
            position = (frameNumber-1)%BUFF_SIZE_OBJECT_DETECTION;

        lhe_advanced_compute_perceptual_relevance (y, pr_x_buff[position], pr_y_buff[position]);

        pr_changes(position);
        
        create_frame(0);

        if (frameNumber > starting_frame) {
            if (is_frame_valid(position)){
                clean_mask();
                add_rectangles_to_mask();
                track_objects();
                find_objects();
            }
            draw_rectangles_in_frame();
        }

        char frameName[100];
        sprintf(frameName,"./output/output%i.bmp",frameNumber);
        if (TRACE_LEVEL>=1) printf("Output: %s\n", frameName);
        yuv420_rgb24_std(width, height, y, u, v, y_stride, uv_stride, rec_rgb, rgb_stride, YCBCR_601);
        stbi_write_bmp(frameName, width, height, rgb_channels, rec_rgb);

    }
    close_pr_computation();
    rectangles_free();

    return 0;
}


