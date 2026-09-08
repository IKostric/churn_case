-- Side-by-side columns for the PDF. Markdown has no column syntax, and
-- pandoc's own `::: columns` support is limited to slide and HTML output, so
-- this maps the same fenced divs onto LaTeX minipages:
--
--   ::: columns
--   :::: {.column width=0.6}
--   ![A caption](figure.png)
--   ::::
--   :::: {.column width=0.36}
--   Text that sits beside the figure.
--   ::::
--   :::
--
-- Widths are fractions of the text width and default to an even split with a
-- gutter. They are not checked, so keep the total below 1.

-- `![caption](img)` becomes a Figure, which is a LaTeX float - and floats are
-- silently dropped inside a minipage. Unwrap it into a plain image plus a
-- \captionof, which numbers identically without floating. The image stays a
-- real pandoc element so that pandoc still resolves it against
-- --resource-path and copies it next to the .tex it compiles.
local function unfloat(blocks)
  local out = pandoc.List()
  for _, block in ipairs(blocks) do
    if block.t ~= 'Figure' then
      out:insert(block)
    else
      out:extend(block.content)
      local caption = block.caption and block.caption.long
      if caption and #caption > 0 then
        -- The argument has to be one raw block; a blank line inside it would
        -- \par and \captionof would fail. Captions are text, so writing them
        -- here costs nothing - unlike the image, which stays an element.
        local inlines = pandoc.Plain(pandoc.utils.blocks_to_inlines(caption))
        local text = pandoc.write(pandoc.Pandoc({ inlines }), 'latex'):gsub('%s+$', '')
        out:insert(pandoc.RawBlock('latex', '\\captionof{figure}{' .. text .. '}'))
      end
    end
  end
  return out
end

function Div(div)
  if not div.classes:includes('columns') then
    return nil
  end

  local columns = pandoc.List()
  for _, block in ipairs(div.content) do
    if block.t == 'Div' and block.classes:includes('column') then
      columns:insert(block)
    end
  end
  if #columns == 0 then
    return nil
  end

  -- an even split, less a 4% gutter between each pair
  local default = string.format('%.4f', (1 - 0.04 * (#columns - 1)) / #columns)
  -- \vspace{0pt} makes [t] align the column tops. Without it the first
  -- baseline of a column holding an image sits at the image's bottom edge, so
  -- a neighbouring text column starts level with the foot of the figure.
  local function open(i)
    return '\\begin{minipage}[t]{' .. (columns[i].attributes.width or default) .. '\\linewidth}%\n'
      .. '\\vspace{0pt}\\setlength{\\parskip}{\\bodyparskip}%'
  end

  local out = pandoc.List({ pandoc.RawBlock('latex', '\\noindent' .. open(1)) })
  for i, column in ipairs(columns) do
    out:extend(unfloat(column.content))
    if i < #columns then
      -- One raw block for the whole join. Pandoc separates blocks with a blank
      -- line, and a \par between \end{minipage} and \hfill\begin{minipage}
      -- would end the horizontal line the columns have to share, stacking them
      -- vertically instead. Blank lines *inside* a minipage are harmless.
      out:insert(pandoc.RawBlock('latex', '\\end{minipage}\\hfill' .. open(i + 1)))
    else
      out:insert(pandoc.RawBlock('latex', '\\end{minipage}'))
    end
  end
  return out
end
